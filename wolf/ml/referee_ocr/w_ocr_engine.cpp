#include "w_ocr_engine.hpp"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <cctype>
#include <cstring>
#include <filesystem>

#include "../w_utilities.hpp"

namespace fs = std::filesystem;

#ifdef __TELEMETRY
#include "opentelemetry/sdk/version/version.h"
#include "opentelemetry/trace/provider.h"

namespace trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

namespace {
nostd::shared_ptr<trace::Tracer> get_tracer() {
  auto provider = trace::Provider::GetTracerProvider();
  return provider->GetTracer("pes_21", OPENTELEMETRY_SDK_VERSION);
}
} // namespace
#endif

using w_ocr_engine = wolf::ml::ocr::w_ocr_engine;
using config_for_ocr_struct = wolf::ml::ocr::config_for_ocr_struct;

w_ocr_engine::w_ocr_engine() {
  std::string key = "TESSERACT_LOG";
  std::string tesseract_log = get_env_string(key.c_str());

  digit_api->Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY);
  digit_api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
  digit_api->SetVariable("tessedit_char_whitelist", "0123456789");
  digit_api->SetVariable("user_defined_dpi", "70");
  digit_api->SetVariable("debug_file", tesseract_log.c_str());

  word_api->Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY);
  word_api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
  word_api->SetVariable(
      "tessedit_char_whitelist",
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"); //	,"ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                                                               //// ,
  word_api->SetVariable("user_defined_dpi", "70");
  word_api->SetVariable("debug_file", tesseract_log.c_str());
}

w_ocr_engine::~w_ocr_engine() {
  digit_api->End();
  word_api->End();
}

bool w_ocr_engine::check_if_overlapped(_In_ cv::Rect p_box_1, _In_ cv::Rect p_box_2,
                                       _In_ config_for_ocr_struct &p_ocr_config) {
  bool if_overlapped = false;
  int area_1, area_2, overlapped_area;

  int dx = std::min(p_box_1.x + p_box_1.width, p_box_2.x + p_box_2.width) -
           std::max(p_box_1.x, p_box_2.x);
  int dy = std::min(p_box_1.y + p_box_1.height, p_box_2.y + p_box_2.height) -
           std::max(p_box_1.y, p_box_2.y);

  if (dx > 0 && dy > 0) {
    area_1 = p_box_1.width * p_box_1.height;
    area_2 = p_box_2.width * p_box_2.height;

    overlapped_area = dx * dy;

    if (double(overlapped_area) / double(area_1) >
            p_ocr_config.overlapped_threshold ||
        double(overlapped_area) / double(area_2) >
            p_ocr_config.overlapped_threshold) {
      if_overlapped = true;
    }
  }

  return if_overlapped;
}

// this function is related to cluster_char_structs function
bool compare_char_by_x_position(
    const w_ocr_engine::characters_struct &p_first_charactor,
    const w_ocr_engine::characters_struct &p_second_charactor) {
  return p_first_charactor.center.x < p_second_charactor.center.x;
}

std::vector<w_ocr_engine::characters_struct>
w_ocr_engine::contours_to_char_structs(
    _In_ std::vector<std::vector<cv::Point>> p_contours) {
  std::vector<characters_struct> modified_contours;
  size_t number_of_contours = p_contours.size();

  for (size_t i = 0; i < number_of_contours; i++) {
    std::vector<cv::Point> contour_poly;
    characters_struct temp_modified_contour;
    temp_modified_contour.contour = p_contours[i];

    double epsilon = 3;
    bool closed = true;
    cv::approxPolyDP(cv::Mat(p_contours[i]), contour_poly, epsilon, closed);

    temp_modified_contour.bound_rect = cv::boundingRect(cv::Mat(contour_poly));

    contour_poly.clear();
    temp_modified_contour.center.x = temp_modified_contour.bound_rect.x +
                                     temp_modified_contour.bound_rect.width / 2;

    temp_modified_contour.center.y =
        temp_modified_contour.bound_rect.y +
        temp_modified_contour.bound_rect.height / 2;

    temp_modified_contour.height = temp_modified_contour.bound_rect.height;

    modified_contours.push_back(temp_modified_contour);
  }
  return modified_contours;
}

void w_ocr_engine::enhance_contour_image_for_model(
    _Inout_ cv::Mat &p_contour_image, _In_ config_for_ocr_struct &p_ocr_config) {
  if (!(p_ocr_config.make_white_background || p_ocr_config.do_resize_contour)) {
    return;
  }

  if (p_ocr_config.make_white_background) {
    make_contour_white_background(p_contour_image, p_ocr_config);
  }

  if (p_ocr_config.do_resize_contour) {
    // float resize_fraction =
    // float(p_ocr_config.desired_contour_height)/float(height);
    int dist_height = p_ocr_config.desired_contour_height;
    int dist_width = 24; // int(resize_fraction*width);

    cv::resize(p_contour_image, p_contour_image, cv::Size(dist_width, dist_height),
               0.0, 0.0, cv::InterpolationFlags::INTER_AREA);
  }

  return;
}

double w_ocr_engine::euclidean_distance(characters_struct &p_first_character,
                                        characters_struct &p_second_character) {
  double dist_x = std::pow(
      float(p_first_character.center.x - p_second_character.center.x), 2.0);
  double dist_y = std::pow(
      float(p_first_character.center.y - p_second_character.center.y), 2.0);
  double dist = std::pow(dist_x + dist_y, 0.5);

  return dist;
}

double w_ocr_engine::euclidean_distance(int p_x1, int p_x2, int p_y1, int p_y2)
{
  double dist_x = std::pow(float(p_x1 - p_x2), 2.0);
  double dist_y = std::pow(float(p_y1 - p_y2), 2.0);
  double dist = std::pow(dist_x + dist_y, 0.5);

  return dist;
}

std::string w_ocr_engine::spaces_between_two_chars(characters_struct p_left_char,
                                            characters_struct p_right_char,
                                            float p_height_to_dist_ratio)
{
  std::string temp_spaces = "";

  int p_left_char_right_corner =
      p_left_char.bound_rect.x + p_left_char.bound_rect.width;
  int p_right_char_left_corner = p_right_char.bound_rect.x;

  if (p_right_char_left_corner - p_left_char_right_corner > 0) {
    if (float(p_right_char_left_corner - p_left_char_right_corner) >
        float(p_left_char.bound_rect.height) * p_height_to_dist_ratio) {
      temp_spaces = "  ";
    } else {
      temp_spaces = " ";
    }
  }

  return temp_spaces;
}

std::vector<w_ocr_engine::character_and_center>
w_ocr_engine::char_clusters_to_text(
    std::vector<std::vector<characters_struct>> p_clustered_characters) {
  std::vector<w_ocr_engine::character_and_center> words;

  for (size_t i = 0; i < p_clustered_characters.size(); i++) {
    std::sort(p_clustered_characters[i].begin(), p_clustered_characters[i].end());
    character_and_center temp;
    temp.center = p_clustered_characters[i][0].center;

    std::string spaces = "";
    float height_to_dist_ratio =
        get_env_float("SOCCER_GLOBAL_HEIGHT_TO_DIST_RATIO");

    for (size_t j = 0; j < p_clustered_characters[i].size(); j++) {
      std::string temp_string =
          split_string(p_clustered_characters[i][j].text, '\n')[0];
      if (j < p_clustered_characters[i].size() - 1) {
        spaces = spaces_between_two_chars(p_clustered_characters[i][j],
                                          p_clustered_characters[i][j + 1],
                                          height_to_dist_ratio);
      }
      temp_string += spaces;
      temp.text.append(temp_string);
    }

    std::transform(temp.text.begin(), temp.text.end(), temp.text.begin(),
                   ::toupper);

    words.push_back(temp);
  }

  p_clustered_characters.clear();

  std::sort(words.begin(), words.end());
  return words;
}

std::vector<w_ocr_engine::characters_struct>
w_ocr_engine::filter_chars_by_contour_size(
    _Inout_ std::vector<characters_struct> &p_character,
    _In_ config_for_ocr_struct &p_ocr_config) {
  std::vector<characters_struct> filtered_characters;
  for (int i = 0; i < p_character.size(); i++) {
    double area = cv::contourArea(p_character[i].contour);
    if (area < p_ocr_config.restrictions.min_area ||
        area > p_ocr_config.restrictions.max_area) {
      continue;
    }
    if (p_character[i].bound_rect.height < p_ocr_config.restrictions.min_height ||
        p_character[i].bound_rect.height > p_ocr_config.restrictions.max_height ||
        p_character[i].bound_rect.width < p_ocr_config.restrictions.min_width ||
        p_character[i].bound_rect.width > p_ocr_config.restrictions.max_width) {
      continue;
    }
    filtered_characters.push_back(p_character[i]);
  }
  return filtered_characters;
}

std::vector<w_ocr_engine::characters_struct>
w_ocr_engine::image_to_char_structs(_In_ cv::Mat &p_image_box,
                                    _In_ config_for_ocr_struct &p_ocr_config) {
  cv::Mat filtered_image =
      prepare_image_for_contour_detection(p_image_box, p_ocr_config);

  std::vector<std::vector<cv::Point>> contours =
      find_all_countors(filtered_image);
  std::vector<characters_struct> characters =
      contours_to_char_structs(contours);

  std::vector<characters_struct> filtered_characters =
      filter_chars_by_contour_size(characters, p_ocr_config);

  merge_overlapped_contours(filtered_characters, p_ocr_config);

  for (size_t i = 0; i < filtered_characters.size(); i++) {
    margin_bounding_rect(filtered_characters[i].bound_rect, p_ocr_config.margin,
                         filtered_image);
  }

  // TODO add this log "This code has not been optimized for color image, yet"
  return filtered_characters;
}

std::vector<w_ocr_engine::character_and_center> w_ocr_engine::char_vec_to_string(
    _In_ std::vector<w_ocr_engine::characters_struct> p_char_vector,
    _In_ cv::Mat &p_frame,
    _In_ config_for_ocr_struct &p_ocr_config)
{
  std::vector<w_ocr_engine::characters_struct> labeled_characters =
      label_chars_in_char_structs(p_char_vector, p_frame, p_ocr_config);
  std::vector<std::vector<w_ocr_engine::characters_struct>>
      clustered_characters =
          cluster_char_structs(labeled_characters, p_ocr_config);
  std::vector<w_ocr_engine::character_and_center> string =
      char_clusters_to_text(clustered_characters);

  return string;
}

std::vector<w_ocr_engine::character_and_center>
w_ocr_engine::image_to_string(_In_ cv::Mat &image,
                              _In_ config_for_ocr_struct &p_ocr_config) {
  std::vector<characters_struct> characters =
      image_to_char_structs(image, p_ocr_config);
  std::vector<characters_struct> labeled_characters =
      label_chars_in_char_structs(characters, image, p_ocr_config);
  std::vector<std::vector<characters_struct>> clustered_characters =
      cluster_char_structs(labeled_characters, p_ocr_config);
  std::vector<character_and_center> string =
      char_clusters_to_text(clustered_characters);

  return string;
}

std::vector<w_ocr_engine::characters_struct>
w_ocr_engine::label_chars_in_char_structs(
    _In_ std::vector<w_ocr_engine::characters_struct> &p_characters,
    _In_ cv::Mat &p_image_box, _In_ config_for_ocr_struct &p_ocr_config) {
  std::vector<characters_struct> labeled_chars;
  tesseract::TessBaseAPI *tess_api;
  if (p_ocr_config.is_digit) {
    tess_api = digit_api;
  } else {
    tess_api = word_api;
  }

  for (size_t i = 0; i < p_characters.size(); i++) {
    cv::Mat temp_contour_image;
    cv::Mat contour_image;
    // Sometimes it is better to use the original image for w_ocr_engine
    if (p_ocr_config.binary) {
      cv::Mat filtered_image =
          prepare_image_for_contour_detection(p_image_box, p_ocr_config);
      filtered_image(p_characters[i].bound_rect).copyTo(contour_image);
    } else {
      // original_image(modified_bounding_rects[i].bound_rect).copyTo(contour_image);
      contour_image = mask_contour(p_image_box, p_characters[i]);
    }

    if (p_ocr_config.is_white) {
      negative_image(contour_image);
    }

    if (p_ocr_config.verbose) {
      temp_contour_image = contour_image.clone();
    }

    enhance_contour_image_for_model(contour_image, p_ocr_config);
    tess_api->SetImage(contour_image.data, contour_image.cols,
                       contour_image.rows, 3, int(contour_image.step));

    std::string text_data = tess_api->GetUTF8Text();

    if (std::strcmp(text_data.c_str(), "") != 0) {
      characters_struct temp_character;
      temp_character = p_characters[i];
      temp_character.text = split_string(text_data, '\n')[0];

      if (!temp_character.text.empty()) {
        temp_character.text = temp_character.text.substr(0, 1);
      }

      labeled_chars.push_back(temp_character);
    }
    contour_image.release();
    temp_contour_image.release();
  }
  return labeled_chars;
}

void w_ocr_engine::margin_bounding_rect(_Inout_ cv::Rect &p_bounding_rect,
                                        _In_ int p_margin,
                                        _In_ cv::Mat &p_filtered_image) {
  int height = p_filtered_image.rows;
  int width = p_filtered_image.cols;
  int temp;
  int temp_margin_width = int(std::ceil(float(p_margin) / 2));
  int temp_margin_height = int(std::ceil(float(p_margin) / 2));

  if (p_bounding_rect.width < 1 && p_margin > 0) {
    temp_margin_width = 2;
  }

  if (p_bounding_rect.width < p_bounding_rect.height / 5) {
    temp = (p_bounding_rect.x - 1 * p_bounding_rect.height / 6);
    p_bounding_rect.x = temp > 0 ? temp : 0;
    temp = (p_bounding_rect.x + p_bounding_rect.height);
    p_bounding_rect.width =
        temp < width ? p_bounding_rect.height / 2 : (width - p_bounding_rect.x - 1);
  } else {
    temp = (p_bounding_rect.x - temp_margin_width);
    p_bounding_rect.x = temp > 0 ? temp : 0;
    temp = (p_bounding_rect.x + p_bounding_rect.width + 3 * temp_margin_width);
    p_bounding_rect.width = temp < width
                              ? p_bounding_rect.width + 3 * temp_margin_width
                              : (width - p_bounding_rect.x - 1);
  }

  temp = (p_bounding_rect.y - temp_margin_height);
  p_bounding_rect.y = temp > 0 ? temp : 0;
  temp = (p_bounding_rect.y + p_bounding_rect.height + 2 * temp_margin_height);
  p_bounding_rect.height = temp < height
                             ? p_bounding_rect.height + 2 * temp_margin_height
                             : (height - p_bounding_rect.y - 1);
}

cv::Mat w_ocr_engine::mask_contour(_In_ cv::Mat &p_image,
                                   _In_ characters_struct &p_contour_info) {
  cv::Mat temp_plane_image =
      cv::Mat(cv::Size(p_image.cols, p_image.rows), CV_8UC1, cv::Scalar(0));
  cv::Mat mask_image;
  cv::Mat contour_image;

  std::vector<std::vector<cv::Point>> temp_contours;
  temp_contours.push_back(p_contour_info.contour);
  std::vector<std::vector<cv::Point>> hull(temp_contours.size());
  for (unsigned int i = 0, n = temp_contours.size(); i < n; ++i) {
    cv::convexHull(cv::Mat(temp_contours[i]), hull[i], false);
  }
  cv::drawContours(temp_plane_image, temp_contours, 0, cv::Scalar(255), 3);
  cv::fillPoly(temp_plane_image, temp_contours, cv::Scalar(255));

  temp_plane_image(p_contour_info.bound_rect).copyTo(mask_image);
  p_image(p_contour_info.bound_rect).copyTo(contour_image, mask_image);

  temp_plane_image.release();
  mask_image.release();
  return contour_image;
}

void w_ocr_engine::merge_overlapped_contours(
    _Inout_ std::vector<characters_struct> &p_character,
    _In_ config_for_ocr_struct &p_ocr_config) {
  cv::Rect ref_box;
  int index;
  std::vector<int> overlapped_boxes_index;
  // int width, height;

  bool flag;
  if (p_character.size() > 0) {
    flag = true;
  } else {
    flag = false;
  }

  index = 0;
  while (flag) {
    ref_box = p_character[index].bound_rect;

    overlapped_boxes_index.clear();
    for (int i = 0; i < p_character.size(); i++) {
      if (i == index) {
        continue;
      }
      if (check_if_overlapped(ref_box, p_character[i].bound_rect, p_ocr_config)) {
        overlapped_boxes_index.push_back(i);
      }
    }

    for (int j = overlapped_boxes_index.size() - 1; j >= 0; j--) {
      p_character[index].bound_rect.x =
          std::min(p_character[index].bound_rect.x,
                   p_character[overlapped_boxes_index[j]].bound_rect.x);
      p_character[index].bound_rect.y =
          std::min(p_character[index].bound_rect.y,
                   p_character[overlapped_boxes_index[j]].bound_rect.y);
      p_character[index].bound_rect.width =
          std::max(p_character[index].bound_rect.x +
                       p_character[index].bound_rect.width,
                   p_character[overlapped_boxes_index[j]].bound_rect.x +
                       p_character[overlapped_boxes_index[j]].bound_rect.width) -
          p_character[index].bound_rect.x;
      p_character[index].bound_rect.height =
          std::max(p_character[index].bound_rect.y +
                       p_character[index].bound_rect.height,
                   p_character[overlapped_boxes_index[j]].bound_rect.y +
                       p_character[overlapped_boxes_index[j]].bound_rect.height) -
          p_character[index].bound_rect.y;
      p_character.erase(p_character.begin() + int(overlapped_boxes_index[j]));
    }

    overlapped_boxes_index.clear();
    if (index >= p_character.size() - 1) {
      flag = false;
    }

    index++;
  }
}

std::vector<std::vector<w_ocr_engine::characters_struct>>
w_ocr_engine::cluster_char_structs(
    std::vector<w_ocr_engine::characters_struct> p_characters,
    config_for_ocr_struct &p_ocr_config) {
  std::vector<std::vector<characters_struct>> clustered_characters;

  if (p_characters.size() == 0) {
    return clustered_characters;
  }

  std::vector<characters_struct> temp_char_cluster;
  std::vector<size_t> temp_index_list;
  bool is_clustering;

  temp_char_cluster.push_back(p_characters.back());
  p_characters.pop_back();

  if (p_characters.size() > 0) {
    is_clustering = true;

    while (is_clustering) {
      for (size_t index = 0; index < p_characters.size(); index++) {
        for (size_t i = 0; i < temp_char_cluster.size(); i++) {
          if (~temp_char_cluster[i].processed) {
            double temp_dist_1 =
                euclidean_distance(temp_char_cluster[i].bound_rect.x,
                                   p_characters[index].bound_rect.x +
                                       p_characters[index].bound_rect.width,
                                   temp_char_cluster[i].bound_rect.y +
                                       temp_char_cluster[i].bound_rect.height,
                                   p_characters[index].bound_rect.y +
                                       p_characters[index].bound_rect.height);
            double temp_dist_2 =
                euclidean_distance(temp_char_cluster[i].bound_rect.x +
                                       temp_char_cluster[i].bound_rect.width,
                                   p_characters[index].bound_rect.x,
                                   temp_char_cluster[i].bound_rect.y +
                                       temp_char_cluster[i].bound_rect.height,
                                   p_characters[index].bound_rect.y +
                                       p_characters[index].bound_rect.height);

            int temp_y_dist = std::abs(temp_char_cluster[i].bound_rect.y -
                                       p_characters[index].bound_rect.y);

            if ((temp_dist_1 < 0.8 * double(temp_char_cluster[i].height) ||
                 temp_dist_2 < 0.8 * double(temp_char_cluster[i].height)) &&
                temp_y_dist < temp_char_cluster[i].height) {
              temp_index_list.push_back(index);
              break;
            }
          }
        }
      }

      for (size_t i = 0; i < temp_char_cluster.size(); i++) {
        temp_char_cluster[i].processed = true;
      }

      if (temp_index_list.size() > 0) {
        std::reverse(temp_index_list.begin(), temp_index_list.end());

        for (size_t i = 0; i < temp_index_list.size(); i++) {
          temp_char_cluster.push_back(p_characters[temp_index_list[i]]);
          p_characters.erase(p_characters.begin() + int(temp_index_list[i]));
        }

        if (p_characters.size() == 0) {
          clustered_characters.push_back(temp_char_cluster);
          temp_char_cluster.clear();
        }
      } else {
        clustered_characters.push_back(temp_char_cluster);
        temp_char_cluster.clear();

        temp_char_cluster.push_back(p_characters.back());
        p_characters.pop_back();
        if (p_characters.size() == 0) {
          clustered_characters.push_back(temp_char_cluster);
          temp_char_cluster.clear();
        }
      }

      if (p_characters.size() == 0) {
        is_clustering = false;
      }

      temp_index_list.clear();
    }
  } else {
    clustered_characters.push_back(temp_char_cluster);
    temp_char_cluster.clear();
    // return;
  }
  return clustered_characters;
}

cv::Mat w_ocr_engine::show_in_better_way(cv::Mat &p_input_image,
                                         int p_out_put_image_height,
                                         float p_resize_factor) {
  int height = p_out_put_image_height;
  int width = height * 4 / 3;
  cv::Mat temp_image;
  if (p_resize_factor * p_input_image.rows > height) {
    p_resize_factor = float(height / p_input_image.rows);
  }
  if (p_resize_factor * p_input_image.cols > width) {
    p_resize_factor = float(width / p_input_image.cols);
  }
  if (p_input_image.channels() == 3) {
    temp_image = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
  } else {
    temp_image = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
  }

  cv::Mat temp_contour = p_input_image.clone();
  cv::resize(temp_contour, temp_contour,
             cv::Size(p_input_image.cols * p_resize_factor,
                      p_input_image.rows * p_resize_factor));
  temp_contour.copyTo(
      temp_image(cv::Rect(0, 0, p_input_image.cols * p_resize_factor,
                          p_input_image.rows * p_resize_factor)));

  temp_contour.release();
  return temp_image;
}

std::vector<std::string> w_ocr_engine::split_string(std::string p_input_string,
                                                    char p_reference) {
  std::stringstream test(p_input_string);
  std::string segment;
  std::vector<std::string> seglist;

  while (std::getline(test, segment, p_reference)) {
    seglist.push_back(segment);
  }

  return seglist;
}

bool w_ocr_engine::same_height(
    _In_ std::vector<characters_struct> p_clustered_chars) {
  bool result = true;
  int average_height = 0;

  for (int i = 0; i < p_clustered_chars.size(); i++) {
    average_height += p_clustered_chars[i].height;
  }
  average_height /= p_clustered_chars.size();
  int min_height = average_height - average_height / 5;
  int max_height = average_height + average_height / 5;

  for (int i = 0; i < p_clustered_chars.size(); i++) {
    if (p_clustered_chars[i].height > max_height ||
        p_clustered_chars[i].height < min_height) {
      result = false;
    }
  }

  return result;
}

bool w_ocr_engine::same_level(
    _In_ std::vector<characters_struct> p_clustered_chars) {
  bool result = true;
  int average_height = 0;
  int average_level = 0;

  for (int i = 0; i < p_clustered_chars.size(); i++) {
    average_height += p_clustered_chars[i].height;
    average_level +=
        p_clustered_chars[i].bound_rect.y + p_clustered_chars[i].bound_rect.height;
  }
  average_height /= p_clustered_chars.size();
  average_level /= p_clustered_chars.size();
  int min_level = average_level - average_height / 10;
  int max_level = average_level + average_height / 10;

  for (int i = 0; i < p_clustered_chars.size(); i++) {
    if (p_clustered_chars[i].bound_rect.y + p_clustered_chars[i].bound_rect.height >
            max_level ||
        p_clustered_chars[i].bound_rect.y + p_clustered_chars[i].bound_rect.height <
            min_level) {
      result = false;
    }
  }

  return result;
}

void w_ocr_engine::show_contours(
    _Inout_ cv::Mat &p_image,
    _In_ std::vector<characters_struct> p_clustered_chars,
    _In_ std::string p_window_name, _In_ bool p_show)
{
  cv::Mat mask_image;

  for (int i = 0; i < p_clustered_chars.size(); i++) {
    std::vector<std::vector<cv::Point>> temp_contours;
    temp_contours.push_back(p_clustered_chars[i].contour);
    std::vector<std::vector<cv::Point>> hull(temp_contours.size());
    for (unsigned int i = 0, n = temp_contours.size(); i < n; ++i) {
      cv::convexHull(cv::Mat(temp_contours[i]), hull[i], false);
    }
    cv::drawContours(p_image, temp_contours, 0, cv::Scalar(255), 3);
    cv::fillPoly(p_image, temp_contours, cv::Scalar(255));
  }

  if (p_show) {
    cv::imshow(p_window_name, p_image);
    cv::waitKey();
  }

  return;
}

w_ocr_engine::cluster_features w_ocr_engine::fill_cluster_features(
    _Inout_ std::vector<characters_struct> &p_clustered_char,
    _In_ int p_image_width, _In_ int p_index) {
  cluster_features features;

  features.index_in_parent_vector = p_index;

  features.min_x = p_clustered_char[0].bound_rect.x;
  features.max_x =
      p_clustered_char[0].bound_rect.x + p_clustered_char[0].bound_rect.width;
  features.min_y = p_clustered_char[0].bound_rect.y;
  features.average_y = p_clustered_char[0].bound_rect.y;
  features.average_height = p_clustered_char[0].bound_rect.height;

  for (int i = 1; i < p_clustered_char.size(); i++) {
    features.min_x = (features.min_x > p_clustered_char[i].bound_rect.x)
                         ? p_clustered_char[i].bound_rect.x
                         : features.min_x;
    features.max_x = (features.max_x > p_clustered_char[i].bound_rect.x +
                                           p_clustered_char[i].bound_rect.width)
                         ? features.max_x
                         : p_clustered_char[i].bound_rect.x +
                               p_clustered_char[i].bound_rect.width;
    features.min_y = (features.min_y > p_clustered_char[i].bound_rect.y)
                         ? p_clustered_char[i].bound_rect.y
                         : features.min_y;
    features.average_y += p_clustered_char[0].bound_rect.y;
    features.average_height += p_clustered_char[0].bound_rect.height;
  }

  features.average_y /= p_clustered_char.size();
  features.average_height /= p_clustered_char.size();

  int mid = p_image_width / 2;
  int temp1 = mid - features.min_x;
  int temp2 = mid - features.max_x;

  if (temp1 >= 0) {
    if (temp2 > 0) {
      features.position = cluster_position::left;
      features.symmetric_x1 = temp2;
      features.symmetric_x2 = temp1;
    } else {
      features.position = cluster_position::middle;
      features.symmetric_x1 = 0;
      features.symmetric_x2 = 0;
    }
  } else {
    if (temp2 >= 0) {
      features.position = cluster_position::middle;
      features.symmetric_x1 = 0;
      features.symmetric_x2 = 0;
    } else {
      features.position = cluster_position::right;
      features.symmetric_x1 = abs(temp1);
      features.symmetric_x2 = abs(temp2);
    }
  }

  return features;
}

bool w_ocr_engine::check_twin_clusters(_In_ cluster_features &p_first_input,
                                       _In_ cluster_features &p_second_input,
                                       _In_ float p_threshold) {
  bool result = false;

  float Y_diff_ration =
      float(std::abs(p_first_input.average_y - p_second_input.average_y)) /
      float((p_first_input.average_height + p_second_input.average_height) / 2);
  float a, b, overlapped_ratio;

  float height_diff_ratio =
      (p_first_input.average_height > p_second_input.average_height)
          ? float(p_first_input.average_height - p_second_input.average_height) /
                float(p_first_input.average_height)
          : float(p_second_input.average_height - p_first_input.average_height) /
                float(p_second_input.average_height);

  if (Y_diff_ration < 0.05 && height_diff_ratio < 0.05) {
    a = (p_first_input.symmetric_x1 < p_second_input.symmetric_x1)
            ? float(p_second_input.symmetric_x1)
            : float(p_first_input.symmetric_x1);
    b = (p_first_input.symmetric_x2 < p_second_input.symmetric_x2)
            ? float(p_first_input.symmetric_x2)
            : float(p_second_input.symmetric_x2);

    overlapped_ratio =
        (b - a) / float(p_first_input.symmetric_x2 - p_first_input.symmetric_x1);
    float temp =
        (b - a) / float(p_second_input.symmetric_x2 - p_second_input.symmetric_x1);

    overlapped_ratio = (overlapped_ratio > temp) ? overlapped_ratio : temp;

    if (overlapped_ratio > p_threshold) {
      result = true;
    }
  }

  return result;
}

void w_ocr_engine::keep_twins(
    _Inout_ std::vector<std::vector<characters_struct>> &p_clustered_char,
    _In_ int p_image_width, _In_ int p_image_height, _In_ bool p_word) {
  std::vector<cluster_features> cluster_features_vector;
  int n_cluster = p_clustered_char.size();

  for (int i = 0; i < n_cluster; i++) {
    cluster_features_vector.push_back(
        fill_cluster_features(p_clustered_char[i], p_image_width, i));
  }

  for (int i = 0; i < n_cluster - 1; i++) {
    if (cluster_features_vector[i].matched) {
      continue;
    }
    for (int j = i + 1; j < n_cluster; j++) {
      if (cluster_features_vector[j].matched) {
        continue;
      }
      if (check_twin_clusters(cluster_features_vector[i],
                              cluster_features_vector[j], 0.6)) {
        cluster_features_vector[i].matched = true;
        cluster_features_vector[j].matched = true;

        cluster_features_vector[i].twin_index_in_parent_vector =
            cluster_features_vector[j].index_in_parent_vector;
        cluster_features_vector[j].twin_index_in_parent_vector =
            cluster_features_vector[i].index_in_parent_vector;
      }
    }
  }

  if (p_word) {
    for (int i = 1; i < n_cluster; i++) {
      if (!cluster_features_vector[i].matched) {
        continue;
      }

      if (cluster_features_vector[i].average_y > p_image_height * 4 / 5) {
        cluster_features_vector[i].matched = false;
        cluster_features_vector[cluster_features_vector[i]
                                    .twin_index_in_parent_vector]
            .matched = false;
      }
    }

    int more = -1;

    for (int i = 1; i < n_cluster; i++) {
      if (!cluster_features_vector[i].matched) {
        continue;
      }
      if (more == -1) {
        more = p_clustered_char[i].size();
      }

      if (more < p_clustered_char[i].size()) {
        more = p_clustered_char[i].size();
      }
    }

    for (int i = 1; i < n_cluster; i++) {
      if (!cluster_features_vector[i].matched) {
        continue;
      }

      if (more > p_clustered_char[i].size() &&
          more > p_clustered_char[cluster_features_vector[i]
                                    .twin_index_in_parent_vector]
                     .size()) {
        cluster_features_vector[i].matched = false;
        cluster_features_vector[cluster_features_vector[i]
                                    .twin_index_in_parent_vector]
            .matched = false;
      }
    }
  } else {
    int largest = -1;

    for (int i = 1; i < n_cluster; i++) {
      if (!cluster_features_vector[i].matched) {
        continue;
      }
      if (largest == -1) {
        largest = cluster_features_vector[i].average_height;
      }

      if (largest < cluster_features_vector[i].average_height) {
        largest = cluster_features_vector[i].average_height;
      }
    }

    for (int i = 1; i < n_cluster; i++) {
      if (!cluster_features_vector[i].matched) {
        continue;
      }
      if (largest == -1) {
        largest = cluster_features_vector[i].average_height;
      }

      if (largest > cluster_features_vector[i].average_height &&
          largest > cluster_features_vector[cluster_features_vector[i]
                                                .twin_index_in_parent_vector]
                        .average_height) {
        cluster_features_vector[i].matched = false;
        cluster_features_vector[cluster_features_vector[i]
                                    .twin_index_in_parent_vector]
            .matched = false;
      } else {
        largest = cluster_features_vector[i].average_height;
      }
    }
  }

  for (int i = 0; i < n_cluster; i++) {
    int index = n_cluster - (i + 1);
    if (!cluster_features_vector[index].matched) {
      p_clustered_char.erase(p_clustered_char.begin() + index);
    }
  }

  return;
}

void w_ocr_engine::keep_time(
    _Inout_ std::vector<std::vector<characters_struct>> &p_clustered_char) {
  int n_cluster = p_clustered_char.size();
  if (n_cluster == 0) {
    return;
  }
  int height = p_clustered_char[n_cluster - 1][0].bound_rect.y;

  for (int i = 1; i < n_cluster; i++) {
    int index = n_cluster - (i + 1);
    if (height < p_clustered_char[index][0].bound_rect.y) {
      p_clustered_char.erase(p_clustered_char.begin() + index);
    } else {
      height = p_clustered_char[index][0].bound_rect.y;
      p_clustered_char.pop_back();
    }
  }
}

void w_ocr_engine::add_text_to_original_image(
    _Inout_ cv::Mat &p_image,
    _In_ std::vector<characters_struct> &p_clustered_char) {
  for (int i = 0; i < p_clustered_char.size(); i++) {
    cv::putText(p_image, p_clustered_char[i].text,
                cv::Point(p_clustered_char[i].bound_rect.x,
                          p_clustered_char[i].bound_rect.y + 100),
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), false);
  }

  return;
}
