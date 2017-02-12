#include "environment.h"
#include "ale_interface.hpp"
#include <glog/logging.h>
#include <iostream>
#include <vector>

namespace fast_dqn {

void pixel2gray(unsigned char pixel, unsigned char& gray);

class ALEEnvironment : public Environment {

public:  
  ALEEnvironment(bool gui, const std::string rom_path) : ale_(gui) {
    ale_.setBool("display_screen", gui);
    ale_.loadROM(rom_path);

    ActionVect av = ale_.getMinimalActionSet();
    for (int i=0; i < av.size(); i++)
      legal_actions_.push_back(static_cast<ActionCode>(av[i]));
  }

  FrameDataSp PreprocessScreen() {
    ALEScreen raw_screen = ale_.getScreen();
    int h = raw_screen.height(), w = raw_screen.width();
    cv::Mat gray_image(h, w, CV_32FC1);
    unsigned char gray;
    for(int i=0;i<h*w;++i) {
      int row = i / w, col = i % w;
      pixel2gray(raw_screen.getArray()[i], gray);
      gray_image.at<float>(row, col) = (float)gray;
    }
    cv::Mat resized_img(kCroppedFrameSize, kCroppedFrameSize, CV_32FC1);
    cv::resize(gray_image, resized_img, cv::Size(kCroppedFrameSize, kCroppedFrameSize));

    auto screen = std::make_shared<FrameData>();
    for(int i=0;i<kCroppedFrameSize;++i) {
      for(int j=0;j<kCroppedFrameSize;++j) {
        uint8_t c = (uint8_t)resized_img.at<float>(i, j);
        (*screen)[i * kCroppedFrameSize + j] = c;
        //std::cout << (int)c << " "; 
      }
    }
    //std::cout << std::endl;
    return screen;
  }

  /*
  FrameDataSp PreprocessScreen() {
    ALEScreen raw_screen = ale_.getScreen();
    size_t rawFrameWidth = raw_screen.width();
    size_t rawFrameHeight = raw_screen.height();
    std::vector<pixel_t> raw_pixels(rawFrameWidth*rawFrameHeight);
    ale_.getScreenGrayscale(raw_pixels);

    auto screen = std::make_shared<FrameData>();
    assert(rawFrameHeight > rawFrameWidth);
    const auto x_ratio = rawFrameWidth / static_cast<double>(kCroppedFrameSize);
    const auto y_ratio = rawFrameHeight / static_cast<double>(kCroppedFrameSize);
    for (auto i = 0; i < kCroppedFrameSize; ++i) {
      for (auto j = 0; j < kCroppedFrameSize; ++j) {
        // for each pixel in new frame, find corresponding region in old one
        const auto first_x = static_cast<int>(std::floor(j * x_ratio));
        const auto last_x = static_cast<int>(std::floor((j + 1) * x_ratio));
        const auto first_y = static_cast<int>(std::floor(i * y_ratio));
        auto last_y = static_cast<int>(std::floor((i + 1) * y_ratio));
        if (last_y >= rawFrameHeight) {
          last_y = rawFrameHeight-1;
        }
        auto x_sum = 0.0;
        auto y_sum = 0.0;
        uint8_t resulting_color = 0.0;
        for (auto x = first_x; x <= last_x; ++x) {
          double x_ratio_in_resulting_pixel = 1.0;
          if (x == first_x) {
            x_ratio_in_resulting_pixel = x + 1 - j * x_ratio;
          } else if (x == last_x) {
            x_ratio_in_resulting_pixel = x_ratio * (j + 1) - x;
          }
          assert(
              x_ratio_in_resulting_pixel >= 0.0 &&
              x_ratio_in_resulting_pixel <= 1.0);
          for (auto y = first_y; y <= last_y; ++y) {
            double y_ratio_in_resulting_pixel = 1.0;
            if (y == first_y) {
              y_ratio_in_resulting_pixel = y + 1 - i * y_ratio;
            } else if (y == last_y) {
              y_ratio_in_resulting_pixel = y_ratio * (i + 1) - y;
            }
            assert(
                y_ratio_in_resulting_pixel >= 0.0 &&
                y_ratio_in_resulting_pixel <= 1.0);
            const auto grayscale =
              raw_pixels[static_cast<int>(y * rawFrameWidth + x)];
            resulting_color +=
                (x_ratio_in_resulting_pixel / x_ratio) *
                (y_ratio_in_resulting_pixel / y_ratio) * grayscale;
          }
        }
        (*screen)[i * kCroppedFrameSize + j] = resulting_color;
      }
    }
    return screen;
  }
  */

  double ActNoop() {
    double reward = 0;
      for (auto i = 0; i < kInputFrameCount && !ale_.game_over(); ++i) {
        reward += ale_.act(PLAYER_A_NOOP);
      }
    return reward;
  }

  double Act(int action) {
    double reward = 0;
      for (auto i = 0; i < kInputFrameCount && !ale_.game_over(); ++i) {
        reward += ale_.act((Action)action);
      }
    return reward;
  }

  void Reset() { 
    ale_.reset_game(); 
  }

  bool EpisodeOver() { 
    return ale_.game_over(); 
  }

  std::string action_to_string(Environment::ActionCode a) { 
    return action_to_string(static_cast<Action>(a)); 
  }

  const ActionVec& GetMinimalActionSet() {
    return legal_actions_;
  }

 private:

  ALEInterface ale_;
  ActionVec legal_actions_;
  
};

EnvironmentSp CreateEnvironment(
    bool gui, const std::string rom_path) {
  return std::make_shared<ALEEnvironment>(gui, rom_path);
}

void SaveCroppedImage(Environment::FrameDataSp fds, std::string filename) {
  std::cout << "Saving " << filename << std::endl;
  const int height = Environment::kCroppedFrameSize, width = Environment::kCroppedFrameSize;
  const int size = height * width;
  CHECK_EQ(size, fds->size());
  cv::Mat gray_image(height, width, CV_32FC1);
  for(int i=0;i<size;++i) {
    int row = i / width, col = i % width;
    float gray = fds->at(i);// / 255.0;
    std::cout << gray << " ";
    gray_image.at<float>(row, col) = gray;
  }
  std::cout << std::endl;
  cv::imwrite(filename, gray_image);
}

const int ntsc_tbl[] = { 
  0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
  0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
  0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
  0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
  0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
  0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
  0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
  0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
  0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
  0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
  0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
  0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
  0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
  0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
  0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
  0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
  0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
  0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
  0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
  0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
  0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
  0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
  0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
  0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
  0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
  0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
  0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
  0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
  0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
  0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
  0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
  0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
};

void pixel2gray(unsigned char pixel, unsigned char& gray) {
  unsigned char r = (ntsc_tbl[pixel] >> 16) & 0xFF;
  unsigned char g = (ntsc_tbl[pixel] >> 8) & 0xFF;
  unsigned char b = (ntsc_tbl[pixel] >> 0) & 0xFF;
  gray = (unsigned char)round(r * 0.2989 + g * 0.5870 + b * 0.1140);
}


}  // namespace fast_dqn
