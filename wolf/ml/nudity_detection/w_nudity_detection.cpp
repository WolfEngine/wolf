#include "w_nudity_detection.hpp"

#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

#include "../w_utilities.hpp"
#include <opencv2/opencv.hpp>

using w_nud_det = wolf::ml::nudet::w_nud_det;

w_nud_det::w_nud_det(_In_ std::string& nudity_detection_model_path) {
  _model = torch::jit::load(nudity_detection_model_path);
  _model.to(torch::kCUDA);
  _model.eval();

  network_warm_up(get_env_int("TEMP_IMAGE_HEIGHT"), get_env_int("TEMP_IMAGE_WIDTH"));
}

w_nud_det::~w_nud_det() = default;

std::vector<float> w_nud_det::nudity_detection(_In_ uint8_t* pImageData, _In_ const int pImageWidth,
                                 _In_ const int pImageHeight, _In_ const int pImageChannels)
{
  std::vector<float> result;

	torch::Tensor tensor_image = torch::from_blob(pImageData, {1, pImageHeight, pImageWidth, pImageChannels}, torch::kByte);

	std::vector<int> first_permute = get_env_vector_of_int("NUDITY_DETECTION_MODEL_FIRST_PERMUTE");

	tensor_image = tensor_image.permute({first_permute[0], first_permute[1], first_permute[2], first_permute[3]});
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);

	// Define normalization constants
	const float kNormalizationMean[] = {0.485f, 0.456f, 0.406f};
	const float kNormalizationStd[] = {0.229f, 0.224f, 0.225f};

	// Create the normalization transform
	auto normalization_transform = torch::data::transforms::Normalize<float>(
		{kNormalizationMean[0], kNormalizationMean[1], kNormalizationMean[2]},
		{kNormalizationStd[0], kNormalizationStd[1], kNormalizationStd[2]});

	// Apply normalization transform to the image tensor
	tensor_image = normalization_transform(tensor_image);

	std::vector<int> second_permute = get_env_vector_of_int("NUDITY_DETECTION_MODEL_SECOND_PERMUTE");

	tensor_image = tensor_image.permute({second_permute[0], second_permute[1], second_permute[2], second_permute[3]});

	tensor_image = tensor_image.to(torch::kCUDA);

	auto output = _model.forward({tensor_image});

	at::Tensor output_tensor;

	if(output.isTensor())
	{
		output_tensor = output.toTensor();
	}
	else
	{
		// get the vector of tensors from the output IValue
		std::vector<at::Tensor> tensor_vector = output.toTensorVector();

		// extract the first tensor from the vector
		output_tensor = tensor_vector[1];
	}

	for (int i = 0; i < output_tensor.sizes()[1]; i++)
	{
		result.push_back(output_tensor[0][i].item<float>());
	}

  return result;
}

void w_nud_det::network_warm_up(_In_ int pHeight, _In_ int pWidth) {
  for (int i = 0; i < 2; i++) {
    cv::Mat const temp_image = cv::Mat(cv::Size(pWidth, pHeight), CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = nudity_detection(temp_image.data, pWidth, pHeight, temp_image.channels());
  }
}

void w_nud_det::accuracy_check(
	_In_ std::string pInfoFilePath)
{
	// w_nud_det* nudity_detection = new w_nud_det(pModelPath);
	std::vector<std::string> info_of_images = {};
	if (std::filesystem::exists(pInfoFilePath))
	{
		info_of_images = read_text_file_line_by_line(pInfoFilePath);
	}
	else
	{
		std::cout << "The path to the text file is not exist!!!" << std::endl;
	}

	std::vector<std::string> image_path_label = {};
	std::string image_path = "";
	cv::Mat image;
	std::vector<float> value = {};
	int number_of_model_outputs = get_env_int("NUDITY_DETECTION_MODEL_NUMBER_OF_MODEL_OUTPUT");
	int label = 0;
	int number_of_correct_decision = 0;
	int number_of_act = 0;
	float model_threshold = get_env_float("NUDITY_DETECTION_MODEL_THRESHOLD");
	for (int i = 0; i < info_of_images.size(); i++)
	{
		image_path_label = split_string(info_of_images[i], ' ');
		if (image_path_label.size() == 2)
		{
			image_path = image_path_label[0];
			label = std::stoi(image_path_label[1]);

			image = cv::imread(image_path, cv::IMREAD_COLOR);
			value = nudity_detection(
				image.data,
				image.cols,
				image.rows,
				image.channels());

			if (label > 0 && label < number_of_model_outputs + 1)
			{
				number_of_act++;
				if (value[label - 1] > model_threshold)
				{
					number_of_correct_decision++;
				}
			}
		}
	}

	float model_accuracy = float(number_of_correct_decision) / float(number_of_act);

	std::cout << "The accuracy of model for this test database is: " << model_accuracy << std::endl;
}
