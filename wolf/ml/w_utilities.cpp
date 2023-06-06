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

std::string get_nearest_string(_In_ std::string pInput, std::string pFilePath)
{
  // LOG_P(w_log_type::W_LOG_INFO, "path to similar strings file: %s",
  // pFilePath);
  std::ifstream similar_strings(pFilePath);
  float threshold = get_env_float("SIMILARITY_THRESHOLD");
  // LOG_P(w_log_type::W_LOG_INFO, "similarity threshold: %f", threshold);
  std::string candidate_string;
  float best_similarity = 0;
  std::string most_similar;

  if (pInput.length() == 0) {
    return pInput;
  }

  while (std::getline(similar_strings, candidate_string)) {
    float similarity =
        normalized_levenshtein_similarity(pInput, candidate_string);
    if (similarity > best_similarity) {
      most_similar = candidate_string;
      best_similarity = similarity;
    }
  }

  if (best_similarity > threshold) {
    return most_similar;
  } else {
    return pInput;
  }
}

std::string get_nearest_string(_In_ std::string pInput,
                        _In_ std::map<std::string, std::string> pMap)

{
  float threshold = get_env_float("SIMILARITY_THRESHOLD_STAT");
  // LOG_P(w_log_type::W_LOG_INFO, "similarity threshold: %f", threshold);
  std::string candidate_string;
  float best_similarity = 0;
  std::string most_similar;

  if (pInput.length() == 0) {
    return pInput;
  }

  for (auto it = pMap.begin(); it != pMap.end(); it++) {
    candidate_string = it->first;
    float similarity =
        normalized_levenshtein_similarity(pInput, candidate_string);
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

std::string get_value_from_json_file_by_key(std::string pJsonFilePath,
                                            std::string pKey) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("get_value_from_json_file_by_key");
#endif
  using namespace rapidjson;

  std::ifstream ifs{pJsonFilePath};
  if (!ifs.is_open()) {
    fs::path cwd = fs::current_path();
    // spdlog::error("current path {}", cwd.string());
    // spdlog::error("Could not open {} file for reading!", pJsonFilePath);
    std::exit(ENOENT);
  }

  IStreamWrapper isw{ifs};

  Document doc{};
  doc.ParseStream(isw);
  std::string out = doc[pKey.c_str()].GetString();

  return out;
}

std::vector<int>
line_of_numbers_in_string_to_vector_of_integers(std::string pVariable) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan(
      "line_of_numbers_in_string_to_vector_of_integers");
#endif
  std::vector<int> result;

  std::vector<std::string> temp = split_string(pVariable, ',');
  for (int i = 0; i < temp.size(); i++) {
    result.push_back(std::stoi(temp[i]));
  }

  return result;
}

float normalized_levenshtein_similarity(_In_ const std::string &s1,
                                       _In_ const std::string &s2){
  const size_t m = s1.size();
  const size_t n = s2.size();
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
    for (auto c1 : s1) {
      costs[0] = i + 1;
      size_t corner = i;
      size_t j = 0;
      for (auto c2 : s2) {
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
  float max = std::max(s1.length(), s2.length());
  float normalized_distance = distance / max;
  return 1 - normalized_distance;
}

std::vector<std::string> read_text_file_line_by_line(_In_ std::string pFilePath)
{
#ifdef __TELEMETRY
  auto span =
      trace::Scope(get_tracer()->StartSpan("read_text_file_line_by_line"));
#endif
  std::vector<std::string> lines;
  std::string line;

  std::ifstream input_file(pFilePath);
  if (!input_file.is_open()) {
    std::cerr << "Could not open the file - '" << pFilePath << "'" << std::endl;
    return lines;
  }

  while (std::getline(input_file, line)) {
    lines.push_back(line);
  }

  input_file.close();
  return lines;
}

bool replace_string(std::string &str, const std::string &from,
                    const std::string &to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

std::vector<std::string> split_string(std::string input_string,
                                      char reference) {
#ifdef __TELEMETRY
  auto scoped_span = trace::Scope(get_tracer()->StartSpan("split_string"));
  // auto scoped_span = get_tracer()->StartSpan("split_string");
#endif
  std::stringstream test(input_string);
  std::string segment;
  std::vector<std::string> seglist;

  while (std::getline(test, segment, reference)) {
    seglist.push_back(segment);
  }

  return seglist;
}

bool string_2_boolean(std::string pVariable) {
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("string_2_boolean");
#endif
  bool result;
  std::transform(pVariable.begin(), pVariable.end(), pVariable.begin(),
                 ::tolower);

  if (pVariable.compare("true") == 0 || pVariable.compare("false") == 0) {
    std::istringstream is(pVariable);
    is >> std::boolalpha >> result;
  } else {
    throw std::runtime_error(
        "Invalid input, the input must be 'true' or 'false' not " + pVariable);
  }

  return result;
}

#ifdef WOLF_ML_OCR
void store_image_in_folder(
    _In_ std::vector<w_referee::match_result_struct> &pVideoResult,
    _In_ std::string pOutputImageFolderPath, _In_ std::string pVideoPath)
{
#ifdef __TELEMETRY
  auto span = get_tracer()->StartSpan("store_image_in_folder");
#endif
  fs::path temp_video_path = pVideoPath;
  std::string temp_name = temp_video_path.filename().string();
  std::string video_name = split_string(temp_name, '.')[0];

  for (size_t i = 0; i < pVideoResult.size(); i++) {
    fs::path out_path = pOutputImageFolderPath + "/" + video_name + "_" +
                        std::to_string(i) + ".png";
    cv::imwrite(out_path.string().c_str(), pVideoResult[i].result_image);
    cv::waitKey(300);
    pVideoResult[i].release();
  }

  return;
}

void write_results_in_file(
    _In_ std::vector<w_referee::match_result_struct> &pVideoResult,
    _In_ std::string pOutputTextPath) {
#ifdef __TELEMETRY
  auto span = trace::Scope(get_tracer()->StartSpan("write_results_in_file"));
#endif
  for (size_t i = 0; i < pVideoResult.size(); i++) {
    if (pVideoResult[i].home_penalty_result.text.compare("") != 0 &&
        pVideoResult[i].away_penalty_result.text.compare("") != 0) {
      write_in_file_append(pOutputTextPath,
                           pVideoResult[i].stat + "," +
                               pVideoResult[i].home_name.text + "," +
                               pVideoResult[i].home_result.text + "," +
                               pVideoResult[i].away_name.text + "," +
                               pVideoResult[i].away_result.text + "," +
                               pVideoResult[i].home_penalty_result.text + "," +
                               pVideoResult[i].away_penalty_result.text + "," +
                               std::to_string(pVideoResult[i].frame_number));
    } else {
      write_in_file_append(pOutputTextPath,
                           pVideoResult[i].stat + "," +
                               pVideoResult[i].home_name.text + "," +
                               pVideoResult[i].home_result.text + "," +
                               pVideoResult[i].away_name.text + "," +
                               pVideoResult[i].away_result.text + "," +
                               std::to_string(pVideoResult[i].frame_number));
    }
  }

  return;
}
#endif // WOLF_ML_OCR

void write_in_file_append(std::string file_path, std::string content) {
#ifdef __TELEMETRY
  auto scoped_span =
      trace::Scope(get_tracer()->StartSpan("write_in_file_append"));
  // auto scoped_span = get_tracer()->StartSpan("write_in_file_append");
#endif

  std::ofstream file;

  file.open(file_path, std::ios_base::app); // append instead of overwrite
  file << content << std::endl;

  file.close();
  return;
}

void write_in_file(std::string file_path, std::string content) {
#ifdef __TELEMETRY
  auto scoped_span =
      trace::Scope(get_tracer()->StartSpan("write_in_file_append"));
  // auto scoped_span = get_tracer()->StartSpan("write_in_file_append");
#endif

  std::ofstream file;

  file.open(file_path); // overwrite
  file << content << std::endl;

  file.close();
  return;
}

bool is_line_contains_variable(const std::string pStr) {
#ifdef __TELEMETRY
  auto span =
      trace::Scope(get_tracer()->StartSpan("is_line_contains_variable"));
#endif
  bool decision = false;
  if (pStr.size() > 0) {
    if (pStr.at(0) != '#' && pStr.size() > 2) {
      decision = true;
    }
  }
  return decision;
}

void set_env(_In_ const char *pDotEnvFilePath) {
#ifdef __TELEMETRY
  auto span = trace::Scope(get_tracer()->StartSpan("set_env"));
#endif
  std::string env_file_path(pDotEnvFilePath);
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

auto get_env_int(_In_ const char *pKey) -> int {
  int value = -1;
  if (const char *env_p = getenv(pKey)) {
    std::string temp(env_p);
    value = std::stoi(temp);
  } else {
    // TODO add log
  }

  return value;
}

float get_env_float(_In_ const char *pKey) {
  float value = -1;
  if (const char *env_p = getenv(pKey)) {
    std::string temp(env_p);
    value = std::stof(temp);
  } else {
    // TODO add log
  }

  return value;
}

bool get_env_boolean(_In_ const char *pKey) {
  bool value = false;
  if (const char *env_p = getenv(pKey)) {
    std::string temp(env_p);
    value = string_2_boolean(temp);
  } else {
    // TODO add log
  }

  return value;
}

std::string get_env_string(_In_ const char *pKey) {
  std::string value;
  if (const char *env_p = getenv(pKey)) {
    value = std::string(env_p);
  } else {
    // TODO add log
  }

  return value;
}

cv::Rect get_env_cv_rect(_In_ const char *pKey) {
  cv::Rect value = cv::Rect(0, 0, 0, 0);
  if (const char *env_p = getenv(pKey)) {
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
	_In_ const char* pKey)
{
	std::vector<int> int_vector ={};
	if (const char* env_p = getenv(pKey))
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
	_In_ const int pKeyValue)
{
	int label = 0;
	switch (pKeyValue)
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
	_In_ const std::string pDirPath)
{
	std::vector<std::string> all_images_path = {};

	for (const auto& entry : fs::directory_iterator(pDirPath))
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
	_In_ const std::string pImageFolderPath,
	_In_ const std::string pLabeledImageTextFile,
	_In_ const std::string pHistory)
{
	std::vector<std::string> all_images_path = {};
	if (std::filesystem::exists(pImageFolderPath))
	{
		all_images_path = images_in_directory(pImageFolderPath);
	}
	else
	{
		std::cout << "The path to images is not exist!!!" << std::endl;
	}

	int n_images = all_images_path.size();

	std::vector<std::string> text_info = {};
	int last_processed_image_number = 0;
	if (std::filesystem::exists(pHistory))
	{
		text_info = read_text_file_line_by_line(pHistory);
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
			pLabeledImageTextFile,
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

std::string get_first_character_of_string(_In_ std::string pStr, _In_ bool pEscape)
{
  if (pEscape || pStr.length() == 0) {
    return pStr;
  }

  char first_char = pStr[0];
  std::string result(1, first_char);

  return result;
}

} // namespace wolf::ml::ocr
