#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
/// Recording-worker only. Does not configure or open any audio hardware.
@interface JRNeuralRecordingFilter : NSObject
- (nullable instancetype)initWithModelsPath:(NSString *)path
    output:(void (^)(int64_t startFrame, NSData *samples))output;
- (void)offerAtFrame:(int64_t)startFrame microphone:(NSData *)microphone
    reference:(NSData *)reference direct:(NSData *)direct;
- (void)finish;
- (void)reset;
@property(nonatomic, readonly) NSDictionary<NSString *, NSNumber *> *metrics;
@property(nonatomic, readonly, nullable) NSString *lastError;
@end
NS_ASSUME_NONNULL_END
