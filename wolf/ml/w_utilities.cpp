#include "w_utilities.hpp"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/sinks/stdout_color_sinks.h"
// #include "spdlog/spdlog.h"
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

namespace wolf::ml {

std::string get_nearest_string(_In_ std::string p_input,
                               _In_ std::string p_file_path)
{
  std::ifstream similar_strings(p_file_path);
  float threshold = get_env_float("SIMILARITY_THRESHOLD");
  std::string candidate_string;
  float best_similarity = 0;
  std::string most_similar;

  if (p_input.length() == 0) {
    return p_input;
  }

  while (std::getline(similar_strings, candidate_string)) {
    float similarity =
        normalized_levenshtein_similarity(p_input, candidate_string);
    if (similarity > best_similarity) {
      most_similar = candidate_string;
      best_similarity = similarity;
    }
  }

  if (best_similarity > threshold) {
    return most_similar;
  } else {
    return p_input;
  }
}

std::string get_nearest_string(_In_ std::string p_input,
                               _In_ std::map<std::string, std::string> p_map)

{
  float threshold = get_env_float("SIMILARITY_THRESHOLD_STAT");
  // LOG_P(w_log_type::W_LOG_INFO, "similarity threshold: %f", threshold);
  std::string candidate_string;
  float best_similarity = 0;
  std::string most_similar;

  if (p_input.length() == 0) {
    return p_input;
  }

  for (auto it = p_map.begin(); it != p_map.end(); it++) {
    candidate_string = it->first;
    float similarity =
        normalized_levenshtein_similarity(p_input, candidate_string);
    if (similarity > best_similarity) {
      most_similar = candidate_string;
      best_similarity = similarity;
    }
  }

  if (best_similarity > threshold) {
    return most_similar;
  } else {
    return "";
  }
}

std::string get_value_from_json_file_by_key(std::string p_json_file_path,
                                            std::string p_key) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("get_value_from_json_file_by_key");
#endif
  using namespace rapidjson;

  std::ifstream ifs{p_json_file_path};
  if (!ifs.is_open()) {
    fs::path cwd = fs::current_path();
    // spdlog::error("current path {}", cwd.string());
    // spdlog::error("Could not open {} file for reading!", p_json_file_path);
    std::exit(ENOENT);
  }

  IStreamWrapper isw{ifs};

  Document doc{};
  doc.ParseStream(isw);
  std::string out = doc[p_key.c_str()].GetString();

  return out;
}

std::vector<int>
line_of_numbers_in_string_to_vector_of_integers(std::string p_variable) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan(
      "line_of_numbers_in_string_to_vector_of_integers");
#endif
  std::vector<int> result;

  std::vector<std::string> temp = split_string(p_variable, ',');
  for (int i = 0; i < temp.size(); i++) {
    result.push_back(std::stoi(temp[i]));
  }

  return result;
}

float normalized_levenshtein_similarity(_In_ const std::string &p_s1,
                                       _In_ const std::string &p_s2){
  const size_t m = p_s1.size();
  const size_t n = p_s2.size();
  int distance;
  if (m == 0 && n == 0)
    return 0;
  else if (m == 0)
    distance = n;
  else if (n == 0)
    distance = m;
  else {
    std::vector<size_t> costs(n + 1);
    std::iota(costs.begin(), costs.end(), 0);
    size_t i = 0;
    for (auto c1 : p_s1) {
      costs[0] = i + 1;
      size_t corner = i;
      size_t j = 0;
      for (auto c2 : p_s2) {
        size_t upper = costs[j + 1];
        costs[j + 1] = (c1 == c2)
                           ? corner
                           : 1 + std::min(std::min(upper, corner), costs[j]);
        corner = upper;
        ++j;
      }
      ++i;
    }
    distance = costs[n];
  }
  float max = std::max(p_s1.length(), p_s2.length());
  float normalized_distance = distance / max;
  return 1 - normalized_distance;
}

std::vector<std::string> read_text_file_line_by_line(_In_ std::string p_file_path)
{
#ifdef __TELEMETRY
  auto span =
      trace::Scope(get_tracer()->StartSpan("read_text_file_line_by_line"));
#endif
  std::vector<std::string> lines;
  std::string line;

  std::ifstream input_file(p_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Could not open the file - '" << p_file_path << "'" << std::endl;
    return lines;
  }

  while (std::getline(input_file, line)) {
    lines.push_back(line);
  }

  input_file.close();
  return lines;
}

bool replace_string(std::string &p_str,
                    const std::string &p_from,
                    const std::string &p_to) {
  size_t start_pos = p_str.find(p_from);
  if (start_pos == std::string::npos)
    return false;
  p_str.replace(start_pos, p_from.length(), p_to);
  return true;
}

std::vector<std::string> split_string(std::string p_input_string,
                                      char p_reference) {
#ifdef __TELEMETRY
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("split_string"));
  // auto scoped_span = get_tracer()->StartSpan("split_string");
#endif
  std::stringstream test(p_input_string);
  std::string segment;
  std::vector<std::string> seglist;

  while (std::getline(test, segment, p_reference)) {
    seglist.push_back(segment);
  }

  return seglist;
}

bool string_2_boolean(std::string p_variable) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("string_2_boolean");
#endif
  bool result;
  std::transform(p_variable.begin(), p_variable.end(), p_variable.begin(),
                 ::tolower);

  if (p_variable.compare("true") == 0 || p_variable.compare("false") == 0) {
    std::istringstream is(p_variable);
    is >> std::boolalpha >> result;
  } else {
    throw std::runtime_error(
        "Invalid input, the input must be 'true' or 'false' not " + p_variable);
  }

  return result;
}

#ifdef WOLF_ML_OCR
void store_image_in_folder(
    _In_ std::vector<w_referee::match_result_struct> &p_video_result,
    _In_ std::string p_output_image_folder_path,
    _In_ std::string p_video_path)
{
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("store_image_in_folder");
#endif
  fs::path temp_video_path = p_video_path;
  std::string temp_name = temp_video_path.filename().string();
  std::string video_name = split_string(temp_name, '.')[0];

  for (size_t i = 0; i < p_video_result.size(); i++) {
    fs::path out_path = p_output_image_folder_path + "/" + video_name + "_" +
                        std::to_string(i) + ".png";
    cv::imwrite(out_path.string().c_str(), p_video_result[i].result_image);
    cv::waitKey(300);
    p_video_result[i].release();
  }

  return;
}

void write_results_in_file(
    _In_ std::vector<w_referee::match_result_struct> &p_video_result,
    _In_ std::string p_output_text_path) {
#ifdef __TELEMETRY
  auto span = trace::Scope(get_tracer()->StartSpan("write_results_in_file"));
#endif
  for (size_t i = 0; i < p_video_result.size(); i++) {
    if (p_video_result[i].home_penalty_result.text.compare("") != 0 &&
        p_video_result[i].away_penalty_result.text.compare("") != 0) {
      write_in_file_append(p_output_text_path,
                           p_video_result[i].stat + "," +
                               p_video_result[i].home_name.text + "," +
                               p_video_result[i].home_result.text + "," +
                               p_video_result[i].away_name.text + "," +
                               p_video_result[i].away_result.text + "," +
                               p_video_result[i].home_penalty_result.text + "," +
                               p_video_result[i].away_penalty_result.text + "," +
                               std::to_string(p_video_result[i].frame_number));
    } else {
      write_in_file_append(p_output_text_path,
                           p_video_result[i].stat + "," +
                               p_video_result[i].home_name.text + "," +
                               p_video_result[i].home_result.text + "," +
                               p_video_result[i].away_name.text + "," +
                               p_video_result[i].away_result.text + "," +
                               std::to_string(p_video_result[i].frame_number));
    }
  }

  return;
}
#endif // WOLF_ML_OCR

void write_in_file_append(std::string p_file_path,
                          std::string p_content) {
#ifdef __TELEMETRY
  auto scoped_span =
      trace::Scope(get_tracer()->StartSpan("write_in_file_append"));
  // auto scoped_span = get_tracer()->StartSpan("write_in_file_append");
#endif

  std::ofstream file;

  file.open(p_file_path, std::ios_base::app); // append instead of overwrite
  file << p_content << std::endl;

  file.close();
  return;
}

void write_in_file(std::string p_file_path,
                   std::string p_content) {
#ifdef __TELEMETRY
  auto scoped_span =
      trace::Scope(get_tracer()->StartSpan("write_in_file_append"));
  // auto scoped_span = get_tracer()->StartSpan("write_in_file_append");
#endif

  std::ofstream file;

  file.open(p_file_path); // overwrite
  file << p_content << std::endl;

  file.close();
  return;
}

bool is_line_contains_variable(const std::string p_str) {
#ifdef __TELEMETRY
  auto span =
      trace::Scope(get_tracer()->StartSpan("is_line_contains_variable"));
#endif
  bool decision = false;
  if (p_str.size() > 0) {
    if (p_str.at(0) != '#' && p_str.size() > 2) {
      decision = true;
    }
  }
  return decision;
}

void set_env(_In_ const char *p_dot_env_file_path) {
#ifdef __TELEMETRY
  auto span = trace::Scope(get_tracer()->StartSpan("set_env"));
#endif
  std::string env_file_path(p_dot_env_file_path);
  auto lines = read_text_file_line_by_line(env_file_path);

  std::vector<std::vector<std::string>> env_vector;
  for (int i = 0; i < lines.size(); i++) {
    if (is_line_contains_variable(lines[i])) {
      env_vector.push_back(split_string(lines[i], '='));
    }
  }

  for (int i = 0; i < env_vector.size(); i++) {
#ifdef _WIN32
    _putenv_s(env_vector[i][0].c_str(), env_vector[i][1].c_str());
#else
    setenv(env_vector[i][0].c_str(), env_vector[i][1].c_str(), 1);
#endif
  }
}

auto get_env_int(_In_ const char *p_key) -> int {
  int value = -1;
  if (const char *env_p = getenv(p_key)) {
    std::string temp(env_p);
    value = std::stoi(temp);
  } else {
    // TODO add log
  }

  return value;
}

float get_env_float(_In_ const char *p_key) {
  float value = -1;
  if (const char *env_p = getenv(p_key)) {
    std::string temp(env_p);
    value = std::stof(temp);
  } else {
    // TODO add log
  }

  return value;
}

bool get_env_boolean(_In_ const char *p_key) {
  bool value = false;
  if (const char *env_p = getenv(p_key)) {
    std::string temp(env_p);
    value = string_2_boolean(temp);
  } else {
    // TODO add log
  }

  return value;
}

std::string get_env_string(_In_ const char *p_key) {
  std::string value;
  if (const char *env_p = getenv(p_key)) {
    value = std::string(env_p);
  } else {
    // TODO add log
  }

  return value;
}

cv::Rect get_env_cv_rect(_In_ const char *p_key) {
  cv::Rect value = cv::Rect(0, 0, 0, 0);
  if (const char *env_p = getenv(p_key)) {
    std::string temp(env_p);
    std::vector<int> int_vect =
        line_of_numbers_in_string_to_vector_of_integers(temp);
    value = cv::Rect(int_vect[0], int_vect[1], int_vect[2], int_vect[3]);
  } else {
    // TODO add log
  }

  return value;
}

std::vector<int> get_env_vector_of_int(
	_In_ const char* p_key)
{
	std::vector<int> int_vector ={};
	if (const char* env_p = getenv(p_key))
	{
		std::string temp(env_p);
		int_vector = line_of_numbers_in_string_to_vector_of_integers(temp);
	}
	else
	{
		// TODO add log
	}

	return int_vector;
}

int map_key_value_to_label(
	_In_ const int p_key_value)
{
	int label = 0;
	switch (p_key_value)
	{
		// 49 and 177 are keyboard key values related to "1"
		case 49:
		case 177:
			label = 1;
			break;
		// 50 and 178 are keyboard key values related to "2"
		case 50:
		case 178:
			label = 2;
			break;
		// 51 and 179 are keyboard key values related to "3"
		case 51:
		case 179:
			label = 3;
			break;
		// 52 and 180 are keyboard key values related to "4"
		case 52:
		case 180:
			label = 4;
			break;
		default:
			label = -1;
			break;
	}
	return label;
}

std::vector<std::string> images_in_directory(
	_In_ const std::string p_dir_path)
{
	std::vector<std::string> all_images_path = {};

	for (const auto& entry : fs::directory_iterator(p_dir_path))
	{
		if (entry.is_regular_file() &&
			(entry.path().extension() == ".jpeg" ||
			 entry.path().extension() == ".jpg" ||
			 entry.path().extension() == ".png"))
		{
			all_images_path.push_back(entry.path().string());
		}
	}

	return all_images_path;
}

void create_labeled_image_text(
	_In_ const std::string p_image_folder_path,
	_In_ const std::string p_labeled_image_text_file,
	_In_ const std::string p_history)
{
	std::vector<std::string> all_images_path = {};
	if (std::filesystem::exists(p_image_folder_path))
	{
		all_images_path = images_in_directory(p_image_folder_path);
	}
	else
	{
		std::cout << "The path to images is not exist!!!" << std::endl;
	}

	int n_images = all_images_path.size();

	std::vector<std::string> text_info = {};
	int last_processed_image_number = 0;
	if (std::filesystem::exists(p_history))
	{
		text_info = read_text_file_line_by_line(p_history);
		last_processed_image_number = std::stoi(text_info[0]);
	}

	cv::Mat image = cv::Mat(cv::Size(100, 100), CV_8UC3, cv::Scalar(0, 0, 0));
	int label = 0;

	for (int i = last_processed_image_number; i < n_images; i++)
	{
		image = cv::imread(all_images_path[i], cv::IMREAD_COLOR);
		cv::imshow("The target image", image);
		int pressed_key = cv::waitKey();
		label = map_key_value_to_label(pressed_key);
		std::string line_info = all_images_path[i] + " " + std::to_string(label);
		write_in_file_append(
			p_labeled_image_text_file,
			line_info);
		std::cout
			<< " -**- : " << line_info << std::endl;
	}

	image.release();
}

std::string get_relative_path_to_root() {
  fs::path cwd = fs::current_path();
  fs::path dot_env_file_path;
  if (cwd.parent_path().filename().compare("build") == 0) {
    dot_env_file_path = "../../";
  } else if (cwd.filename().compare("build") == 0 ||
             cwd.filename().compare("ocr") == 0) {
    dot_env_file_path = "../";
  } else if (cwd.parent_path().parent_path().filename().compare("build") == 0) {
    dot_env_file_path = "../../../";
  } else {
    dot_env_file_path = "";
  }

  std::string temp = dot_env_file_path.string();

  return temp;
}

std::string get_first_character_of_string(_In_ std::string p_str, _In_ bool p_scape)
{
  if (p_scape || p_str.length() == 0) {
    return p_str;
  }

  char first_char = p_str[0];
  std::string result(1, first_char);

  return result;
}

} // namespace wolf::ml::ocr
