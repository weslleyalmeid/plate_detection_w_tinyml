#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

typedef cv::Point3_<float> Pixel;
template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);


void normalize(Pixel &pixel){
    pixel.x = ((pixel.x - 255.0) / 255.0);
    pixel.y = ((pixel.y - 255.0) / 255.0);
    pixel.z = ((pixel.z - 255.0) / 255.0);
}

template<>
float* TensorData(TfLiteTensor* tensor, int batch_index) {
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
    switch (tensor->type) {
        case kTfLiteFloat32:
            return tensor->data.f + nelems * batch_index;
    }
    return nullptr;
}

std::vector<std::string> load_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    if (!file.is_open())
    {
        fprintf(stderr, "unable to open label file\n");
        exit(-1);
    }
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str))
    {
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }
    file.close();
    return labels;
}

int main(int argc, char **argv)
{

    // Get Model label and input image
    if (argc != 4)
    {
        fprintf(stderr, "TfliteClassification.exe modelfile labels image\n");
        exit(-1);
    }
    // const char *modelFileName = argv[1];
    // const char *labelFile = argv[2];
    // const char *imageFile = argv[3];

    const char *modelFileName = "../models/plate/float32_320_320.tflite";
    // const char *modelFileName = "../models/classificao/mobilenet_v1_1.0_224_quant.tflite";
    const char *imageFile = "../images/BAG-7751.jpg";

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // Initiate Interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }
    // Configure the interpreter
    // interpreter->SetAllowFp16PrecisionForFp32(true);
    // interpreter->SetNumThreads(1);
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    
    // ################ METODO ANTIGO ##############
    auto frame = cv::imread(imageFile);
    if (false){
    // Load Input Image
        cv::Mat image;

        auto frame = cv::imread(imageFile);
        if (frame.empty())
        {
            fprintf(stderr, "Failed to load iamge\n");
            exit(-1);
        }

        // Copy image to input tensor
        cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), image.data, image.rows * image.cols * image.elemSize());
    }
    // ################ METODO NOVO 1 - - image shape (1, 320, 320, 3) ############## 
   
    
    if (true){
        
        cv::Mat image;
        frame.convertTo(image, CV_32FC3);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        Pixel* pixel = image.ptr<Pixel>(0,0);
        const Pixel* endPixel = pixel + image.cols * image.rows;
        
        for (; pixel != endPixel; pixel++)
            normalize(*pixel);

        cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);

        cv::imwrite("../images/salvei_agora_3.png", image);

        float* inputImg_ptr = image.ptr<float>(0);
        float* inputLayer = interpreter->typed_input_tensor<float>(0);

        memcpy(inputLayer, inputImg_ptr, width * height * 2 * sizeof(float));
    }

    // ################ METODO NOVO 2 - - image shape (1, 320, 320, 3) ############## 
    if (false){
        cv::Mat image = cv::imread(imageFile);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);

        // for (size_t i = 0; size_t < image.size(); i++) {
        //     const auto& rgb = image[i];
        //     interpreter->typed_input_tensor<unsigned char>(0)[3*i + 0] = rgb[0];
        //     interpreter->typed_input_tensor<unsigned char>(0)[3*i + 1] = rgb[1];
        //     interpreter->typed_input_tensor<unsigned char>(0)[3*i + 2] = rgb[2];
        // }
    }

    // Inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // ######### MEU OUTPUT ##########
    // Get count
    std::vector<std::pair<float, int>> result_count;
    float threshold_count = 0.01f;
    float* outputLayer = interpreter->typed_output_tensor<float>(0);

    int count_tensor = interpreter->outputs()[0];
    TfLiteIntArray *output_dims_count = interpreter->tensor(count_tensor)->dims;
    auto count = output_dims_count->data[output_dims_count->size - 1];

    tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), count, 1, threshold_count, &result_count, kTfLiteFloat32);

    // Get boxes
    int box_tensor = interpreter->outputs()[1];
    TfLiteIntArray *output_dims_box = interpreter->tensor(box_tensor)->dims;
    auto boxes = output_dims_box->data[output_dims_box->size - 1];

    // Get classes
    int classes_tensor = interpreter->outputs()[2];
    TfLiteIntArray *output_dims_classes = interpreter->tensor(classes_tensor)->dims;
    auto classes = output_dims_classes->data[output_dims_classes->size - 1];

    // Get scores
    int scores_tensor = interpreter->outputs()[3];
    TfLiteIntArray *output_dims_scores = interpreter->tensor(scores_tensor)->dims;
    auto scores = output_dims_scores->data[output_dims_scores->size - 1];

    // int outputs = interpreter->outputs();

    // int input = interpreter->outputs()[0];
    // const int    num_detections    = *TensorData<float>(outputs[3], 0);
    // const float* detection_classes =  TensorData<float>(outputs[1], 0);
    // const float* detection_scores  =  TensorData<float>(outputs[2], 0);
    // const float* detection_boxes   =  TensorData<float>(outputs[0], 0);

    // ######### ORIGINAL ##########

    // Get Output
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    
    std::vector<std::pair<float, int>> result;
    float threshold = 0.01f;

    switch (interpreter->tensor(output)->type)
    {
    case kTfLiteInt32:
        tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size, 1, threshold, &result, kTfLiteFloat32);
        break;
    case kTfLiteFloat32:
        tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size, 1, threshold, &result, kTfLiteFloat32);
        break;
    case kTfLiteUInt8:
        tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &result, kTfLiteUInt8);
        break;
    default:
        fprintf(stderr, "cannot handle output type\n");
        exit(-1);
    }
    // Print inference ms in input image
    cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    // Load Labels

    // Print labels with confidence in input image
    std::string output_txt = "camera";
    cv::putText(frame, output_txt, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    // Display image
    cv::imshow("Output", frame);
    cv::waitKey(0);

    return 0;
}


