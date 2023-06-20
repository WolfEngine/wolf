/*
    Project: Wolf Engine. Copyright © 2014-2023 Pooya Eimandar
    https://github.com/WolfEngine/WolfEngine
*/

#pragma once

#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

#include "wolf.hpp"

namespace wolf::ml::nudet {

class w_nud_det {
 public:
  /*!
          The constructor of the class.
  */
  explicit w_nud_det(_In_ std::string& nudity_detection_model_path);

  /*!
          The deconstructor of the class.
  */
  ~w_nud_det();

  /*!
  The nudity_detection function accepts image information as input and returns a float vector 
  containing the model result.

          \param p_image_data the spacial image pixel data.
          \param p_image_width the image width.
          \param p_image_height the image height.
          \param p_image_channels the number of image channels.
          \return a vector of float numbers each between 0 to 1 that shows the nudity factors.
  */
  W_API std::vector<float> nudity_detection(_In_ uint8_t* p_image_data,
                                _In_ int p_image_width, _In_ int p_image_height, _In_ int p_image_channels);

  /*!
  The function uses to warm-up the network in the w_nud_det class initialization.

          \param p_height the temp image height.
          \param p_width the temp image width.
          \return (void)
  */
  W_API void network_warm_up(_In_ int p_height, _In_ int p_width);

  /*!
  The function uses to calculate the accuracy of the input model over pre-labeled images.  

  	\param p_info_file_path the path to the labeled file.
  	\return the accuracy of ml model throw given picturs.
  */
  W_API float accuracy_check(
  	_In_ std::string p_info_file_path);

 private:
  torch::jit::script::Module _model;
};

}  // namespace wolf::ml::nudet
