/*
    Project: Wolf Engine. Copyright © 2014-2023 Pooya Eimandar
    https://github.com/WolfEngine/WolfEngine
*/

#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef WOLF_ML_OCR
#include "referee_ocr/w_referee.hpp"
#endif // WOLF_ML_OCR
#include "wolf.hpp"

#ifdef WOLF_ML_OCR
using w_referee = wolf::ml::ocr::w_referee;
#endif // WOLF_ML_OCR

namespace wolf::ml {

/*!
        The get_nearest_string returns the nearest string to input among strings
   stored to the file specified by p_file_path environment variable. when the most
   similar string is found, its similarity is compared to SIMILARITY_THRESHOLD
   environment variable and if greater than it, the similar string is returned.
   otherwise, the input string is returned.

        \param p_input The input string.
        \param p_file_path The file path contains target strings.
        \return The most similar string to input string.
*/
W_API std::string get_nearest_string(_In_ std::string p_input,
                              _In_ std::string p_file_path);

/*!
        return the nearest string to input among strings stored in the input
   p_map. when the most similar string is found, its similarity is compared to
   SIMILARITY_THRESHOLD environment variable and if greater than it, the similar
   string is returned. otherwise, the input string is returned.

        \param p_input The input string.
        \param p_map A map contains the target strings.
        \return most similar string to input string.
*/
W_API std::string get_nearest_string(_In_ std::string p_input,
                              _In_ std::map<std::string, std::string> p_map);

/*!
        The function gets the specific value by it's key and return the value in
   string format.

        \param p_json_file_path the path of json file.
        \param p_key the corresponding key that related to the desired value.
        \return desired value.
*/
W_API std::string get_value_from_json_file_by_key(std::string p_json_file_path,
                                                  std::string p_key);

/*!
        The function gets a string as input, the string contains numbers of int
   numbers separated by spaces and split them and return them in a vector of
   integers.

        \param p_variable string of integers number.
        \return a vector of integers.
*/
W_API std::vector<int>
line_of_numbers_in_string_to_vector_of_integers(std::string p_variable);

/*!
        compute the normalized similarity between input strings using the
   Levenshtein metric. output is a number between 0 and 1. 1 shows high
   similarity and 0 shows low similarity.

        \param p_s1 first string.
        \param p_s2 second string.
        \return normalized similarity metric.
*/
W_API float normalized_levenshtein_similarity(_In_ const std::string &p_s1,
                                             _In_ const std::string &p_s2);

/*!
        replace the specified phrase with another specified phrase in string.

        \param p_str input string.
        \param p_from first phrase.
        \param p_to second phrase.
        \return boolean parameter show success or failure of function
*/
W_API bool replace_string(std::string &p_str,
                          const std::string &p_from,
                          const std::string &p_to);

/*!
        The function gets a string as input and return the boolean
   representation of the input.

        \param p_variable the string input.
        \return the boolean representain of the input string.
*/
W_API bool string_2_boolean(std::string p_variable);

#ifdef WOLF_ML_OCR
/*!
        The function gets a string as input, the string contains numbers of int
   numbers separated by spaces and split them and return them in a vector of
   integers.

        \param p_video_result The result of the video.
        \param p_output_image_folder_path The path for the output image folder.
        \param p_video_path The path of the game video.
        \return
*/
W_API void store_image_in_folder(
    _In_ std::vector<w_referee::match_result_struct> &p_video_result,
    _In_ std::string p_output_image_folder_path, _In_ std::string p_video_path);

/*!
        The function stores the video output text result in the p_output_text_path
   file path.

        \param p_video_result The result of the video.
        \param p_output_text_path The path for the output text file.
        \return
*/
W_API void write_results_in_file(
    _In_ std::vector<w_referee::match_result_struct> &p_video_result,
    _In_ std::string p_output_text_path);
#endif // WOLF_ML_OCR

W_API void write_in_file_append(std::string p_file_path, std::string p_content);

W_API void write_in_file(std::string p_file_path, std::string p_content);
W_API std::vector<std::string> split_string(std::string p_input_string,
                                            char p_reference);

/*!
        The function reads all lines of the input file and returns them in a
   string vector.

        \param p_file_path The path of the input file.
        \return a vector of strings.
*/
W_API std::vector<std::string> read_text_file_line_by_line(_In_ std::string p_file_path);

/*!
        The .env file may have empty or commented lines. The
   is_line_contains_variable functions use to detect these lines.

        \param p_str The input string.
        \return False, if the input string contains # or is empty.
*/
W_API bool is_line_contains_variable(const std::string p_str);

/*!
        The function reads environment variables from the .env file and set them
   in the environment by using the putenv function.

        \param p_dot_env_file_path The path of the .env file.
        \return
*/
W_API void set_env(_In_ const char *p_dot_env_file_path);

/*!
        The function return the value of an environment variable based on the
   input key.

        \param p_key The path of the .env file.
        \return the value of the variable in int.
*/
W_API int get_env_int(_In_ const char *p_key);

/*!
        The function return the value of an environment variable based on the
   input key.

        \param p_key The path of the .env file.
        \return the value of the variable in float.
*/
W_API float get_env_float(_In_ const char *p_key);

/*!
        The function return the value of an environment variable based on the
   input key.

        \param p_key The path of the .env file.
        \return the value of the variable in boolean.
*/
W_API bool get_env_boolean(_In_ const char *p_key);

/*!
        The function return the value of an environment variable based on the
   input key.

        \param p_key The path of the .env file.
        \return the value of the variable in string.
*/
W_API std::string get_env_string(_In_ const char *p_key);

/*!
        The function return the value of an environment variable based on the
   input key.

        \param p_key The path of the .env file.
        \return The value of the variable in cv::Rect.
*/
W_API cv::Rect get_env_cv_rect(_In_ const char *p_key);

/*!
	The function return the value of an environment variable based on the input key.

	\param p_key The path of the .env file.
	\return the value of the variable in vector<int>.
*/
W_API std::vector<int> get_env_vector_of_int(
	_In_ const char* p_key);

/*!
	The function takes an input image folder path and client tags as parameters. It
	then generates a text file that contains the path to each image in the folder
	and its corresponding label based on the client tags. This output file can be
	used for further analysis or training of machine learning models. Additionally,
	the function stores the last processed image number in a separate history text
	file, allowing the function to resume from where it left off if it is interrupted
	or stopped before processing all images in the folder.

	\param p_image_folder_path the image file path.
	\param p_labeled_image_text_file the p_labeled_image_text_file file path.
	\param p_history the file that stores the last processed image number.
	\return (void)
*/
W_API void create_labeled_image_text(
	_In_ std::string p_image_folder_path,
	_In_ std::string p_labeled_image_text_file,
	_In_ std::string p_history);

/*!
	The function return the value of an environment variable based on the input key.

	\param p_dir_path The path of image folder.
	\return the result would be a vector contains image paths
*/
W_API std::vector<std::string> images_in_directory(
	_In_ const std::string p_dir_path);


/*!
        The function returns the related root path compared to the current path.

        \return The related root path compared to the current path.
*/
W_API std::string get_relative_path_to_root();

/*!
        The get_first_character_of_string function returns the first character
   of the string.

        \param p_str The input string.
        \param p_scape if true then return input string without any change.
        \return The first character of input string.
*/
W_API std::string get_first_character_of_string(_In_ std::string p_str,
                                         _In_ bool p_scape);
} // namespace wolf::ml::ocr
