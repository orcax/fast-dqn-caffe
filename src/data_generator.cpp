#include "fast_dqn.h"
#include "environment.h"
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cmath>
#include <iostream>
#include <deque>
#include <algorithm>
#include <boost/filesystem.hpp>

DEFINE_bool(verbose, false, "verbose output");
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_int32(gpu_id, 0, "GPU ID");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(game, "breakout", "Atari 2600 ROM to play");
DEFINE_string(solver, "models/dqn_solver.prototxt", "Solver parameter"
  "file (*.prototxt)");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon"
  "to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory_threshold, 100, "Enough amount of transitions to start "
  "learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_string(model, "model/dqn_iter_2500000.caffemodel", "Model file to load");
//DEFINE_string(model, "", "Model file to load");
DEFINE_bool(evaluate, true, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_double(repeat_games, 2, "Number of games played in evaluation mode");
DEFINE_int32(steps_per_epoch, 5000, "Number of training steps per epoch");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(fast_dqn::EnvironmentSp environmentSp,
    fast_dqn::Fast_DQN* dqn, const double epsilon, const std::string dir) {
  assert(!environmentSp->EpisodeOver());
  std::deque<fast_dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !environmentSp->EpisodeOver(); ++frame) {
    if (FLAGS_verbose)
      LOG(INFO) << "frame: " << frame;
    const auto current_frame = environmentSp->PreprocessScreen();
//     if (FLAGS_show_frame) {
//       std::cout << fast_dqn::DrawFrame(*current_frame);
//     }
    past_frames.push_back(current_frame);
    if (past_frames.size() < fast_dqn::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      environmentSp->ActNoop();
    } else {
      if (past_frames.size() > fast_dqn::kInputFrameCount) {
        past_frames.pop_front();
      }
      fast_dqn::State input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      const auto action = dqn->SelectAction(input_frames, epsilon);

      auto immediate_score = environmentSp->Act(action);
      total_score += immediate_score;

      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      auto reward = immediate_score;

      // save images
      std::stringstream ss;
      ss << dir << "/" << std::setw(5) << std::setfill('0') << frame << ".jpg";
      fast_dqn::Environment::FrameDataSp fds = environmentSp->PreprocessScreen();
      fast_dqn::SaveCroppedImage(fds, ss.str());
    }
  }
  environmentSp->Reset();
  return total_score;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(FLAGS_gpu_id);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  std::string rom = "roms/" + FLAGS_game + ".bin";
  fast_dqn::EnvironmentSp environmentSp = fast_dqn::CreateEnvironment(FLAGS_gui, rom);

  // Get the vector of legal actions
  const fast_dqn::Environment::ActionVec legal_actions = environmentSp->GetMinimalActionSet();

  fast_dqn::Fast_DQN dqn(environmentSp, legal_actions, FLAGS_solver,
      FLAGS_memory, FLAGS_gamma, FLAGS_verbose);

  dqn.Initialize();

  CHECK(!FLAGS_model.empty());
  LOG(INFO) << "Loading " << FLAGS_model;
  dqn.LoadTrainedModel(FLAGS_model);

  auto mkdir = [=](std::string path) {
    boost::filesystem::path dir(path);
    return boost::filesystem::create_directory(dir);
  };

  auto total_score = 0.0;
  for (auto ep = 0; ep < FLAGS_repeat_games; ++ep) {
    LOG(INFO) << "game: " << ep << " ";

    std::stringstream ss;
    ss << "data/";
    mkdir(ss.str());
    ss << FLAGS_game << "/";
    mkdir(ss.str());
    ss << std::setw(4) << std::setfill('0') << ep;
    mkdir(ss.str());
    const auto score = PlayOneEpisode(environmentSp, &dqn, FLAGS_evaluate_with_epsilon, ss.str());

    LOG(INFO) << "score: " << score;
    total_score += score;
    environmentSp->Reset();
  }
  LOG(INFO) << "average score: " << total_score / FLAGS_repeat_games;

  return 0;
}

