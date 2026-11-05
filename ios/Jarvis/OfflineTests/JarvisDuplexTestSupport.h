#import "../Jarvis/Services/JarvisDuplexAudioIO.h"

NS_ASSUME_NONNULL_BEGIN
@interface JRDTestDelegate : NSObject <RTCAudioDeviceDelegate>
@property(nonatomic, readonly) NSInteger notificationViolations;
@property(nonatomic, readonly) NSInteger parameterNotifications;
@property(nonatomic, readonly) NSInteger interruptionNotifications;
@end
void JRDTestReset(void);
void JRDTestSetFailure(int stage);
int JRDTestStarts(void);
int JRDTestAllocations(void);
int JRDTestDisposals(void);
void JRDTestPump(UInt32 frames, UInt64 hostTime);
BOOL JRDTestLastInputWasSilent(void);
BOOL JRDTestLastOutputWasSilent(void);
int JRDRunLowLevelChecks(void);
NS_ASSUME_NONNULL_END
