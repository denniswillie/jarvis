#pragma once

// Recording-only neural echo cancellation; no live audio hardware ownership.
// DTLN inference adapted from breizhn/DTLN-aec; see DTLN-LICENSE.
#include <Accelerate/Accelerate.h>
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace recording_aec {
constexpr int Rate=16000, Block=128, Window=512, History=35328;
using Samples=std::array<float,Block>;
using Feature=std::array<double,6>;
inline void require(bool okay,const char* reason) { if(!okay) throw std::runtime_error(reason); }

class Fourier {
    vDSP_DFT_Setup forward_=vDSP_DFT_zop_CreateSetup(nullptr,Window,vDSP_DFT_FORWARD);
    vDSP_DFT_Setup inverse_=vDSP_DFT_zop_CreateSetup(nullptr,Window,vDSP_DFT_INVERSE);
public:
    Fourier() { require(forward_ && inverse_,"DFT setup failed"); }
    Fourier(const Fourier&)=delete;
    Fourier& operator=(const Fourier&)=delete;
    ~Fourier() { if(forward_)vDSP_DFT_DestroySetup(forward_); if(inverse_)vDSP_DFT_DestroySetup(inverse_); }
    void forward(const float* input,float* re,float* im) const {
        std::array<float,Window> zero{};
        vDSP_DFT_Execute(forward_,input,zero.data(),re,im);
    }
    void inverse(const float* re,const float* im,float* output) const {
        std::array<float,Window> imaginary{};
        vDSP_DFT_Execute(inverse_,re,im,output,imaginary.data());
        for(int i=0;i<Window;++i)output[i]/=Window;
    }
};

class Network {
    std::unique_ptr<TfLiteModel,decltype(&TfLiteModelDelete)> model_{nullptr,TfLiteModelDelete};
    std::unique_ptr<TfLiteDelegate,decltype(&TfLiteXNNPackDelegateDelete)> delegate_{nullptr,TfLiteXNNPackDelegateDelete};
    std::unique_ptr<TfLiteInterpreterOptions,decltype(&TfLiteInterpreterOptionsDelete)> options_{nullptr,TfLiteInterpreterOptionsDelete};
    std::unique_ptr<TfLiteInterpreter,decltype(&TfLiteInterpreterDelete)> interpreter_{nullptr,TfLiteInterpreterDelete};
public:
    std::vector<float> state;
    explicit Network(const std::string& path) {
        model_.reset(TfLiteModelCreateFromFile(path.c_str())); require(bool(model_),"model open failed");
        auto d=TfLiteXNNPackDelegateOptionsDefault(); d.num_threads=1;
        delegate_.reset(TfLiteXNNPackDelegateCreate(&d)); require(bool(delegate_),"XNNPACK setup failed");
        options_.reset(TfLiteInterpreterOptionsCreate()); require(bool(options_),"interpreter options failed");
        TfLiteInterpreterOptionsSetNumThreads(options_.get(),1);
        TfLiteInterpreterOptionsAddDelegate(options_.get(),delegate_.get());
        interpreter_.reset(TfLiteInterpreterCreate(model_.get(),options_.get()));
        require(bool(interpreter_),"interpreter creation failed");
        require(TfLiteInterpreterAllocateTensors(interpreter_.get())==kTfLiteOk,"tensor allocation failed");
        require(TfLiteInterpreterGetInputTensorCount(interpreter_.get())==3 &&
                TfLiteInterpreterGetOutputTensorCount(interpreter_.get())==2,"unexpected model interface");
        state.resize(TfLiteTensorByteSize(TfLiteInterpreterGetInputTensor(interpreter_.get(),1))/sizeof(float));
    }
    void input(int index,const float* data,size_t count) {
        auto* tensor=TfLiteInterpreterGetInputTensor(interpreter_.get(),index);
        require(TfLiteTensorType(tensor)==kTfLiteFloat32 && TfLiteTensorByteSize(tensor)==count*sizeof(float),"input shape mismatch");
        require(TfLiteTensorCopyFromBuffer(tensor,data,count*sizeof(float))==kTfLiteOk,"input copy failed");
    }
    void output(int index,float* data,size_t count) {
        auto* tensor=TfLiteInterpreterGetOutputTensor(interpreter_.get(),index);
        require(TfLiteTensorByteSize(tensor)==count*sizeof(float),"output shape mismatch");
        require(TfLiteTensorCopyToBuffer(tensor,data,count*sizeof(float))==kTfLiteOk,"output copy failed");
    }
    void invoke() {
        input(1,state.data(),state.size());
        require(TfLiteInterpreterInvoke(interpreter_.get())==kTfLiteOk,"model inference failed");
        output(1,state.data(),state.size());
    }
};

class Model {
    Fourier fft_;
    Network first_,second_;
    std::array<float,Window> mic_{},ref_{},sum_{};
public:
    explicit Model(const std::string& directory):first_(directory+"/dtln_aec_512_1.tflite"),
        second_(directory+"/dtln_aec_512_2.tflite") {}
    void reset() {
        mic_.fill(0);ref_.fill(0);sum_.fill(0);
        std::fill(first_.state.begin(),first_.state.end(),0);
        std::fill(second_.state.begin(),second_.state.end(),0);
    }
    Samples process(const Samples& mic,const Samples& ref) {
        std::move(mic_.begin()+Block,mic_.end(),mic_.begin());
        std::move(ref_.begin()+Block,ref_.end(),ref_.begin());
        std::copy(mic.begin(),mic.end(),mic_.end()-Block);
        std::copy(ref.begin(),ref.end(),ref_.end()-Block);
        std::array<float,Window> mr{},mi{},rr{},ri{},estimate{},decoded{};
        std::array<float,257> mm{},rm{},mask{};
        fft_.forward(mic_.data(),mr.data(),mi.data());fft_.forward(ref_.data(),rr.data(),ri.data());
        for(int i=0;i<257;++i){mm[i]=std::hypot(mr[i],mi[i]);rm[i]=std::hypot(rr[i],ri[i]);}
        first_.input(0,mm.data(),mm.size());first_.input(2,rm.data(),rm.size());first_.invoke();first_.output(0,mask.data(),mask.size());
        for(int i=0;i<Window;++i){float gain=mask[i<=256?i:Window-i];mr[i]*=gain;mi[i]*=gain;}
        fft_.inverse(mr.data(),mi.data(),estimate.data());
        second_.input(0,estimate.data(),estimate.size());second_.input(2,ref_.data(),ref_.size());second_.invoke();second_.output(0,decoded.data(),decoded.size());
        std::move(sum_.begin()+Block,sum_.end(),sum_.begin());std::fill(sum_.end()-Block,sum_.end(),0);
        for(int i=0;i<Window;++i)sum_[i]+=decoded[i];
        Samples out{};std::copy_n(sum_.begin(),Block,out.begin());return out;
    }
};

class Alignment {
    Fourier fft_;
    std::array<float,Window> mic_{},ref_{};
    std::deque<Feature> micFeatures_,refFeatures_;
    std::deque<float> history_;
    int64_t lastProbe_=0,lastLock_=-1;
    int pending_=-1;
    Feature feature(const std::array<float,Window>& samples) {
        std::array<float,Window> windowed{},re{},im{};
        for(int i=0;i<Window;++i)windowed[i]=samples[i]*float(.5-.5*std::cos(2*M_PI*i/(Window-1)));
        fft_.forward(windowed.data(),re.data(),im.data());
        constexpr int edges[]={100,250,500,1000,2000,4000,7900};
        Feature out{};
        for(int b=0;b<6;++b){double sum=0;int count=0;
            for(int i=0;i<257;++i){double hz=double(i)*Rate/Window;if(hz>=edges[b] && hz<edges[b+1]){sum+=double(re[i])*re[i]+double(im[i])*im[i];++count;}}
            out[b]=std::log(std::max(sum/std::max(count,1),1e-12));}
        return out;
    }
public:
    int64_t end=0;
    int lag=-1;
    int accepted=0,rejected=0;
    void clear(){mic_.fill(0);ref_.fill(0);micFeatures_.clear();refFeatures_.clear();history_.clear();lag=-1;pending_=-1;lastLock_=-1;lastProbe_=end;}
    bool push(const Samples& mic,const Samples& ref) {
        std::move(mic_.begin()+Block,mic_.end(),mic_.begin());std::move(ref_.begin()+Block,ref_.end(),ref_.begin());
        std::copy(mic.begin(),mic.end(),mic_.end()-Block);std::copy(ref.begin(),ref.end(),ref_.end()-Block);
        for(float v:ref){history_.push_back(v);if(history_.size()>History)history_.pop_front();}
        end+=Block;
        if(history_.size()>=Window){micFeatures_.push_back(feature(mic_));refFeatures_.push_back(feature(ref_));
            if(micFeatures_.size()>273){micFeatures_.pop_front();refFeatures_.pop_front();}}
        if(history_.size()<35200 || end-lastProbe_<2048)return false;
        lastProbe_=end;
        constexpr int count=32,maximum=150;
        std::array<Feature,count> x{};Feature xm{},xn{};
        for(int i=0;i<count;++i)for(int b=0;b<6;++b)xm[b]+=micFeatures_[micFeatures_.size()-count+i][b]/count;
        for(int i=0;i<count;++i)for(int b=0;b<6;++b){x[i][b]=micFeatures_[micFeatures_.size()-count+i][b]-xm[b];xn[b]+=x[i][b]*x[i][b];}
        std::array<double,maximum+1> scores{};
        for(int k=0;k<=maximum;++k){Feature ym{},yn{},dot{};
            size_t base=refFeatures_.size()-count-maximum+k;
            for(int i=0;i<count;++i)for(int b=0;b<6;++b)ym[b]+=refFeatures_[base+i][b]/count;
            for(int i=0;i<count;++i)for(int b=0;b<6;++b){double y=refFeatures_[base+i][b]-ym[b];yn[b]+=y*y;dot[b]+=y*x[i][b];}
            for(int b=0;b<6;++b)scores[k]+=dot[b]/std::max(std::sqrt(xn[b]*yn[b]),1e-9)/6;}
        int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());double score=scores[best],second=-1;
        for(int k=0;k<=maximum;++k)if(std::abs(k-best)>2)second=std::max(second,scores[k]);
        double energy=0;for(float v:history_)energy+=double(v)*v;
        int proposed=(maximum-best)*Block;
        bool good=score>=.55 && second<score*.98 && energy/history_.size()>1e-10;
        if(good && (lag<0 || std::abs(proposed-lag)>.064*Rate)){bool confirmed=pending_>=0 && std::abs(proposed-pending_)<=2*Block;pending_=proposed;good=confirmed;}
        else if(good)pending_=-1;else pending_=-1;
        if(!good){++rejected;return false;}
        bool reset=lag<0 || std::abs(proposed-lag)>.05*Rate;lag=proposed;lastLock_=end;++accepted;return reset;
    }
    bool usable() const {
        if(lag<0 || lastLock_<0 || end-lastLock_>4*Rate || history_.empty())return false;
        double energy=0;size_t n=std::min(history_.size(),size_t(Rate));
        for(size_t i=history_.size()-n;i<history_.size();++i)energy+=double(history_[i])*history_[i];
        return energy/n>1e-10;
    }
    Samples reference(int bulkFrames) const {
        Samples out{};int64_t first=end-int64_t(history_.size());
        for(int i=0;i<Block;++i){int64_t at=end-Block+i-bulkFrames;if(at>=first && at<end)out[i]=history_[size_t(at-first)];}
        return out;
    }
};

enum class Mode { disabled, fixed, automatic };
struct Output { int64_t startFrame; Samples microphone; };
struct Stats { int64_t inputs=0,outputs=0,filtered=0,fallback=0;int resets=0;double modelSeconds=0; };

class Worker {
    struct Pending { int64_t start;Samples raw;bool usable,forceRaw; };
    Mode mode_;
    std::unique_ptr<Model> model_;
    Alignment alignment_;
    std::deque<Pending> pending_;
    int sinceReset_=0;
    float wet_=0;
    bool closed_=false,wasValid_=true;
    std::function<void(const Output&)> sink_;
    Samples inference(const Samples& mic,const Samples& ref){
        if(!model_)return Samples{};
        auto t=std::chrono::steady_clock::now();auto out=model_->process(mic,ref);
        stats.modelSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return out;
    }
    void emit(const Samples& clean){auto q=pending_.front();pending_.pop_front();
        float target=q.usable && sinceReset_>3?1:0;if(q.forceRaw || sinceReset_<=3)wet_=0;
        Output out{q.start,{}};
        for(int i=0;i<Block;++i){float weight=wet_+(target-wet_)*float(i)/(Block-1);out.microphone[i]=q.raw[i]*(1-weight)+clean[i]*weight;}
        wet_=target;stats.outputs+=Block;(target?stats.filtered:stats.fallback)+=Block;sink_(out);
    }
public:
    Stats stats;
    Worker(const std::string& models,Mode mode,std::function<void(const Output&)> sink):mode_(mode),sink_(std::move(sink)){
        if(mode!=Mode::disabled)model_=std::make_unique<Model>(models);
        if(mode==Mode::fixed){for(int i=0;i<3;++i)model_->process(Samples{},Samples{});}
    }
    void process(int64_t start,const Samples& mic,const Samples& ref,bool referenceValid=true){
        require(!closed_ && start==stats.inputs,"noncontiguous or closed recording worker");
        for(int i=0;i<Block;++i)require(std::isfinite(mic[i])&&std::isfinite(ref[i]),"nonfinite samples");
        bool reset=false;
        if(referenceValid)reset=alignment_.push(mic,ref);
        else{alignment_.clear();alignment_.end+=Block;reset=wasValid_;}
        wasValid_=referenceValid;
        if(reset && mode_==Mode::automatic){model_->reset();sinceReset_=0;++stats.resets;}
        bool usable=referenceValid && (mode_==Mode::fixed || (mode_==Mode::automatic && alignment_.usable()));
        int bulk=mode_==Mode::fixed?8960:std::max(0,alignment_.lag-640);
        Samples rendered=referenceValid?alignment_.reference(bulk):Samples{};
        auto clean=inference(mic,rendered);++sinceReset_;
        pending_.push_back({start,mic,usable,!referenceValid});stats.inputs+=Block;
        if(pending_.size()>3)emit(clean);
    }
    void finish(){if(closed_)return;while(!pending_.empty()){auto out=inference(Samples{},Samples{});++sinceReset_;emit(out);}closed_=true;}
    int lastLag() const{return alignment_.lag;}
    int accepted() const{return alignment_.accepted;}
};
} // namespace recording_aec
