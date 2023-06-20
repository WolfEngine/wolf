#include "w_image_processor.hpp"

#include "salieri.h"

namespace wolf::ml::ocr {

std::vector<std::vector<cv::Point>>
find_all_countors(_In_ cv::Mat &p_filtered_image) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(p_filtered_image, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  return contours;
}

void gaussian_blur(_Inout_ cv::Mat &p_frame_box,
                   _In_ config_for_ocr_struct &p_ocr_config) {
  int kernel_size = p_ocr_config.gaussian_blur_win_size;
  cv::GaussianBlur(p_frame_box, p_frame_box, cv::Size(kernel_size, kernel_size), 0,
                   0);
}

void make_contour_white_background(_Inout_ cv::Mat &p_contour_image,
                                   _In_ config_for_ocr_struct &p_ocr_config) {
  int height = p_contour_image.rows;
  int width = p_contour_image.cols;
  int n_channels = p_contour_image.channels();

  if (p_ocr_config.make_white_background) {
    if (n_channels == 1) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (p_contour_image.at<uchar>(i, j) >
              p_ocr_config.white_background_threshold) {
            p_contour_image.at<uchar>(i, j) = 180;
          }
        }
      }
    } else if (n_channels == 3) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          cv::Vec3b &color_pixel = p_contour_image.at<cv::Vec3b>(i, j);
          int count = 0, combine = 0;
          for (int i = 0; i < 3; i++) {
            if (color_pixel[i] > p_ocr_config.white_background_threshold) {
              count++;
              combine += color_pixel[i];
            }
          }
          if (count > 2 ||
              combine > p_ocr_config.white_background_threshold * 2 + 100) {
            color_pixel[0] =
                255; // (color_pixel[0]*2 > 255) ? 255:color_pixel[0];
            color_pixel[1] =
                255; // (color_pixel[1]*2 > 255) ? 255:color_pixel[1];
            color_pixel[2] =
                255; // (color_pixel[2]*2 > 255) ? 255:color_pixel[2];
          }
        }
      }
    }
  }
}

void negative_image(_Inout_ cv::Mat &p_contour_image) {
  int height = p_contour_image.rows;
  int width = p_contour_image.cols;
  int n_channels = p_contour_image.channels();

  if (n_channels == 1) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        p_contour_image.at<uchar>(i, j) = 255 - p_contour_image.at<uchar>(i, j);
      }
    }
  } else if (n_channels == 3) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        cv::Vec3b &color_pixel = p_contour_image.at<cv::Vec3b>(i, j);
        color_pixel[0] = 255 - color_pixel[0];
        color_pixel[1] = 255 - color_pixel[1];
        color_pixel[2] = 255 - color_pixel[2];
      }
    }
  }
}

cv::Mat
prepare_image_for_contour_detection(_In_ cv::Mat &p_image,
                                    _In_ config_for_ocr_struct &p_ocr_config) {
  cv::Mat filtered_image = p_image.clone();
  if (p_ocr_config.do_resize) {
    int dist_height = p_ocr_config.resized_height;
    resize_image(filtered_image, dist_height);
  }

  if (p_ocr_config.do_blur) {
    gaussian_blur(filtered_image, p_ocr_config);
  }

  if (p_ocr_config.do_threshold) {
    threshold_image(filtered_image, p_ocr_config);
  }
  return filtered_image;
}

void resize_image(_Inout_ cv::Mat &p_frame_box, _In_ int p_dst_height,
                  _In_ int p_dst_width) {
  /*!<fraction = 1.0*/
  float ratio;
  cv::Size dim;

  int frame_height = p_frame_box.rows;
  int frame_width = p_frame_box.cols;

  if (p_dst_width == -1 && p_dst_height == -1) {
    return;
  }

  if (p_dst_width == -1) {
    ratio = float(p_dst_height) / float(frame_height);
    dim.width = int(frame_width * ratio);
    dim.height = p_dst_height;
    // fraction = r;
  } else {
    ratio = float(p_dst_width) / float(frame_width);
    dim.width = p_dst_width;
    dim.height = int(frame_height * ratio);
    // fraction = r;
  }
  cv::resize(p_frame_box, p_frame_box, dim);
}

void threshold_image(_Inout_ cv::Mat &p_frame_box,
                     _In_ config_for_ocr_struct &p_ocr_config) {
  if (p_frame_box.channels() == 3) {
    if (p_ocr_config.binary) {
      cv::cvtColor(p_frame_box, p_frame_box, cv::COLOR_BGR2GRAY);
      cv::threshold(p_frame_box, p_frame_box, 0, 255, cv::THRESH_OTSU);
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3),
                                                 cv::Point(0, 0));

      cv::morphologyEx(p_frame_box, p_frame_box, cv::MORPH_CLOSE, kernel);
    } else {
      cv::inRange(p_frame_box,
                  cv::Scalar(p_ocr_config.threshold_value,
                             p_ocr_config.threshold_value,
                             p_ocr_config.threshold_value),
                  cv::Scalar(255, 255, 255), p_frame_box);
    }
  } else if (p_frame_box.channels() == 1) {
    cv::threshold(p_frame_box, p_frame_box, p_ocr_config.threshold_value, 255,
                  cv::THRESH_BINARY);
  }
  // return image;
}
} // namespace wolf::ml::ocr
