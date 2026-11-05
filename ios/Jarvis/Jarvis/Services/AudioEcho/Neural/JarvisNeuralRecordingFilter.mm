#import "JarvisNeuralRecordingFilter.h"
#import <AVFoundation/AVFoundation.h>
#include "NativeRecordingAEC.hpp"

namespace {
using Audio = std::vector<float>;
using Sink = std::function<void(int64_t, const Audio&)>;
using recording_aec::require;

class Converter {
    AVAudioConverter *converter_;
    AVAudioFormat *inputFormat_;
    AVAudioFormat *outputFormat_;
public:
    Converter(double from, double to) {
        inputFormat_ = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:from channels:1];
        outputFormat_ = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:to channels:1];
        converter_ = [[AVAudioConverter alloc] initFromFormat:inputFormat_ toFormat:outputFormat_];
        require(converter_ != nil, "sample-rate converter unavailable");
        converter_.primeMethod = AVAudioConverterPrimeMethod_Normal;
    }
    Audio process(const float *data, size_t count, bool finish = false) {
        AVAudioPCMBuffer *input = nil;
        if (count) {
            input = [[AVAudioPCMBuffer alloc] initWithPCMFormat:inputFormat_ frameCapacity:(AVAudioFrameCount)count];
            require(input != nil, "input buffer unavailable");
            input.frameLength = (AVAudioFrameCount)count;
            std::copy(data, data + count, input.floatChannelData[0]);
        }
        __block bool supplied = false;
        Audio result;
        for (int drain = 0; drain < 32; ++drain) {
            AVAudioPCMBuffer *output = [[AVAudioPCMBuffer alloc] initWithPCMFormat:outputFormat_ frameCapacity:8192];
            NSError *error = nil;
            auto status = [converter_ convertToBuffer:output error:&error withInputFromBlock:
                ^AVAudioBuffer *(AVAudioPacketCount requested, AVAudioConverterInputStatus *inputStatus) {
                    if (input && !supplied) {
                        supplied = true;
                        *inputStatus = AVAudioConverterInputStatus_HaveData;
                        return input;
                    }
                    *inputStatus = finish ? AVAudioConverterInputStatus_EndOfStream : AVAudioConverterInputStatus_NoDataNow;
                    return nil;
                }];
            require(error == nil && status != AVAudioConverterOutputStatus_Error, "sample-rate conversion failed");
            if (output.frameLength)
                result.insert(result.end(), output.floatChannelData[0], output.floatChannelData[0] + output.frameLength);
            if (status == AVAudioConverterOutputStatus_EndOfStream || status == AVAudioConverterOutputStatus_InputRanDry)
                return result;
            require(output.frameLength > 0, "sample-rate converter stalled");
        }
        throw std::runtime_error("sample-rate converter drain limit");
    }
};

struct Totals {
    int64_t inputs=0, outputs=0, fallback=0, segments=0, failures=0, peakPending=0, blocks=0;
    int64_t workerInputs=0, workerOutputs=0, workerFiltered=0, workerBypassed=0;
    int64_t alignmentUpdates=0, modelResets=0, lastLag=-1;
    double processingSeconds=0, maxOfferSeconds=0;
};

void addWorkerMetrics(Totals& totals, const recording_aec::Worker& worker) {
    totals.workerInputs += worker.stats.inputs;
    totals.workerOutputs += worker.stats.outputs;
    totals.workerFiltered += worker.stats.filtered;
    totals.workerBypassed += worker.stats.fallback;
    totals.alignmentUpdates += worker.accepted();
    totals.modelResets += worker.stats.resets;
    totals.lastLag = worker.lastLag();
}

class Pipeline {
    std::string models_;
    Sink sink_;
    bool active_=false, failed_=false;
    int64_t start_=0, expected_=0, emitted_=0, workerInput_=0;
    std::deque<float> raw_, direct_, mic16_, ref16_;
    std::unique_ptr<Converter> micDown_, refDown_, cleanUp_;
    std::unique_ptr<recording_aec::Worker> worker_;

    void emit(const Audio& clean, bool fallback = false) {
        const size_t count = std::min(clean.size(), raw_.size());
        if (!count) return;
        Audio mixed(count);
        for (size_t i=0; i<count; ++i) {
            const float value = clean[i] + direct_.front();
            require(std::isfinite(value), "nonfinite filtered audio");
            mixed[i] = std::clamp(value, -1.0f, 1.0f);
            direct_.pop_front(); raw_.pop_front();
        }
        sink_(start_ + emitted_, mixed);
        emitted_ += count; totals.outputs += count;
        if (fallback) totals.fallback += count;
    }
    void emitRaw() {
        Audio remainder(raw_.begin(), raw_.end());
        emit(remainder, true);
    }
    void append16(std::deque<float>& queue, const Audio& values) {
        queue.insert(queue.end(), values.begin(), values.end());
    }
    void pump(bool final) {
        require(mic16_.size() == ref16_.size(), "reference and microphone conversion diverged");
        while (mic16_.size() >= recording_aec::Block || (final && !mic16_.empty())) {
            recording_aec::Samples mic{}, ref{};
            size_t count = std::min(mic16_.size(), size_t(recording_aec::Block));
            for (size_t i=0; i<count; ++i) {
                mic[i]=mic16_.front(); ref[i]=ref16_.front();
                mic16_.pop_front(); ref16_.pop_front();
            }
            worker_->process(workerInput_, mic, ref);
            workerInput_ += recording_aec::Block; ++totals.blocks;
        }
    }
    void release() {
        if (worker_) addWorkerMetrics(totals, *worker_);
        worker_.reset(); micDown_.reset(); refDown_.reset(); cleanUp_.reset();
        mic16_.clear(); ref16_.clear(); active_=false;
    }
    void fail(const std::exception& error) {
        NSLog(@"Recording neural filter bypass: %s", error.what());
        lastError = std::string(error.what()).substr(0, 512);
        failed_=true; ++totals.failures;
        emitRaw(); release();
    }
    void initialize() {
        micDown_=std::make_unique<Converter>(48000,16000);
        refDown_=std::make_unique<Converter>(48000,16000);
        cleanUp_=std::make_unique<Converter>(16000,48000);
        worker_=std::make_unique<recording_aec::Worker>(models_, recording_aec::Mode::automatic,
            [this](const recording_aec::Output& out) {
                emit(cleanUp_->process(out.microphone.data(), out.microphone.size()));
            });
    }
public:
    Totals totals;
    std::string lastError;
    Totals metrics() const {
        auto result = totals;
        if (worker_) addWorkerMetrics(result, *worker_);
        return result;
    }
    Pipeline(std::string models, Sink sink): models_(std::move(models)), sink_(std::move(sink)) {}
    void offer(int64_t start, const float *mic, const float *ref, const float *direct, size_t count) {
        auto begin=std::chrono::steady_clock::now();
        if (!count) return;
        if (active_ && start != expected_) finish();
        bool first=!active_;
        if (first) { start_=start; expected_=start; emitted_=0; workerInput_=0; active_=true; ++totals.segments; }
        raw_.insert(raw_.end(), mic, mic+count);
        direct_.insert(direct_.end(), direct, direct+count);
        expected_ += count; totals.inputs += count;
        totals.peakPending=std::max(totals.peakPending, int64_t(raw_.size()));
        try {
            require(raw_.size() <= 96000, "recording filter queue limit");
            for (size_t i=0;i<count;++i)
                require(std::isfinite(mic[i]) && std::isfinite(ref[i]) && std::isfinite(direct[i]), "nonfinite input audio");
            if (failed_) emitRaw();
            else {
                if (first) initialize();
                append16(mic16_, micDown_->process(mic,count));
                append16(ref16_, refDown_->process(ref,count));
                pump(false);
            }
        } catch (const std::exception& error) { fail(error); }
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        totals.processingSeconds += elapsed;
        totals.maxOfferSeconds=std::max(totals.maxOfferSeconds,elapsed);
    }
    void finish() {
        if (!active_) return;
        try {
            if (worker_) {
                append16(mic16_,micDown_->process(nullptr,0,true));
                append16(ref16_,refDown_->process(nullptr,0,true));
                pump(true); worker_->finish();
                emit(cleanUp_->process(nullptr,0,true));
            }
            // Any converter shortfall preserves the original microphone tail.
            emitRaw(); release();
        } catch (const std::exception& error) { fail(error); }
    }
};
}

@implementation JRNeuralRecordingFilter {
    std::unique_ptr<Pipeline> _pipeline;
}
- (instancetype)initWithModelsPath:(NSString *)path output:(void (^)(int64_t, NSData *))output {
    self = [super init];
    if (!self) return nil;
    for (NSString *name in @[@"dtln_aec_512_1.tflite", @"dtln_aec_512_2.tflite"]) {
        if (![[NSFileManager defaultManager] isReadableFileAtPath:[path stringByAppendingPathComponent:name]]) {
            NSLog(@"Recording neural filter model unavailable");
            return nil;
        }
    }
    void (^callback)(int64_t, NSData *) = [output copy];
    _pipeline=std::make_unique<Pipeline>(path.UTF8String, [callback](int64_t start,const Audio& values) {
        callback(start,[NSData dataWithBytes:values.data() length:values.size()*sizeof(float)]);
    });
    return self;
}
- (void)offerAtFrame:(int64_t)start microphone:(NSData *)mic reference:(NSData *)ref direct:(NSData *)direct {
    if (mic.length != ref.length || mic.length != direct.length || mic.length % sizeof(float)) {
        [NSException raise:NSInvalidArgumentException format:@"Recording audio arrays must have equal float32 lengths"];
    }
    _pipeline->offer(start,static_cast<const float *>(mic.bytes),static_cast<const float *>(ref.bytes),
                    static_cast<const float *>(direct.bytes),mic.length/sizeof(float));
}
- (void)finish { _pipeline->finish(); }
- (void)reset { _pipeline->finish(); }
- (NSDictionary<NSString *,NSNumber *> *)metrics {
    const auto t=_pipeline->metrics();
    return @{@"input_frames":@(t.inputs),@"output_frames":@(t.outputs),@"fallback_frames":@(t.fallback),
             @"segments":@(t.segments),@"failures":@(t.failures),@"peak_pending_frames":@(t.peakPending),
             @"model_blocks":@(t.blocks),@"processing_seconds":@(t.processingSeconds),@"max_offer_seconds":@(t.maxOfferSeconds),
             @"worker_input_frames_16k":@(t.workerInputs),@"worker_output_frames_16k":@(t.workerOutputs),
             @"worker_filtered_frames_16k":@(t.workerFiltered),@"worker_bypassed_frames_16k":@(t.workerBypassed),
             @"alignment_updates":@(t.alignmentUpdates),@"model_resets":@(t.modelResets),
             @"last_alignment_ms":@(t.lastLag < 0 ? -1.0 : t.lastLag * 1000.0 / recording_aec::Rate)};
}
- (NSString *)lastError {
    return _pipeline->lastError.empty() ? nil : @(_pipeline->lastError.c_str());
}
@end
