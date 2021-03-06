#ifndef SRC_ENVIRONMENT_H_
#define SRC_ENVIRONMENT_H_
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace fast_dqn {

  // Abstract environment class
  // implementation must define the class 
  //    EnvironmentSp CreateEnvironment( bool gui, const std::string rom_path);

class Environment;
typedef std::shared_ptr<Environment> EnvironmentSp;

class Environment {
 public:
  typedef std::vector<int> ActionVec;
  typedef int ActionCode;

  static constexpr auto kRawFrameHeight = 250;
  static constexpr auto kRawFrameWidth = 160;
  static constexpr auto kCroppedFrameSize = 84;
  static constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
  static constexpr auto kInputFrameCount = 4;
  static constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;

  using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
  using FrameDataSp = std::shared_ptr<FrameData>;
  using State = std::array<FrameDataSp, kInputFrameCount>;

  virtual FrameDataSp PreprocessScreen() = 0;

  virtual double ActNoop() = 0;

  //virtual double Act(int Action) = 0;
  virtual double Act(int act_idx) = 0;

  virtual void Reset() = 0;

  virtual bool EpisodeOver() = 0;

  //virtual std::string action_to_string(ActionCode a) = 0;
  virtual std::string action_to_string(int act_idx) = 0;

  //virtual const ActionVec& GetMinimalActionSet() = 0;
  virtual const int num_acts() = 0; // return number of actions

};

// Factory method
EnvironmentSp CreateEnvironment(bool gui, const std::string rom_path);

void SaveCroppedImage(Environment::FrameDataSp fds, std::string filename);

}  // namespace fast_dqn

#endif  // SRC_ENVIRONMENT_H_
