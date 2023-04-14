#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

typedef cv::Point3_<float> Pixel;

void normalize(Pixel &pixel){
    pixel.x = ((pixel.x - 255.0) / 255.0);
    pixel.y = ((pixel.y - 255.0) / 255.0);
    pixel.z = ((pixel.z - 255.0) / 255.0);
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
    // const char *imageFile = "../images/BAG-7751.jpg";
    const char *imageFile = "../images/input/EUL-0433.jpg";
    

    // ##################### Load Model #####################
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // ##################### Initiate Interpreter #####################
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

    // ##################### Configure the interpreter #####################
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(3);

    // ##################### Get Input Tensor Dimensions #####################
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];


    // ##################### Adjust Image shape (1, 320, 320, 3) ##################### 

    auto frame = cv::imread(imageFile);
    cv::Mat image;
    frame.convertTo(image, CV_32FC3);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    Pixel* pixel = image.ptr<Pixel>(0,0);
    const Pixel* endPixel = pixel + image.cols * image.rows;
    
    for (; pixel != endPixel; pixel++)
        normalize(*pixel);

    cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);
    float* inputImg_ptr = image.ptr<float>(0);
    float* inputLayer = interpreter->typed_input_tensor<float>(0);

    memcpy(inputLayer, inputImg_ptr, width * height * channels * sizeof(float));


    // ##################### Inference ##################### 
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


    std::vector<std::pair<float, int>> result;
    float threshold = 0.01f;

    auto box_tensor = interpreter->outputs()[1];
    TfLiteTensor *output_box = interpreter->tensor(box_tensor);
    auto output_boxes = output_box->data.f;
    // std::vector<float> boxes;
    std::vector<float> boxes;

    for (int i = 0; i < 40; i++){
        auto output = output_boxes[i];
        boxes.push_back(output);
	}

    // Get classes
    auto classes_tensor = interpreter->outputs()[3];
    TfLiteTensor *output_class = interpreter->tensor(classes_tensor);
    auto output_classes = output_class->data.f;
    std::vector<float> classes;

    for (int i = 0; i < 10; i++){
        classes.push_back(output_classes[i]);
	}

    // Get scores
    auto scores_tensor = interpreter->outputs()[0];
    TfLiteTensor *output_score = interpreter->tensor(scores_tensor);
    auto output_scores = output_score->data.f;
    std::vector<float> scores;

    for (int i = 0; i < 10; i++){
        scores.push_back(output_scores[i]);
	}

    // Get count
    auto count_tensor = interpreter->outputs()[2];
    TfLiteTensor *output_count = interpreter->tensor(count_tensor);
    auto output_counts = output_count->data.f;


    // ##################### Save image ##################### 

    float CAMERA_WIDTH = width;
    float CAMERA_HEIGHT = height;

    float ymin_temp = boxes[0];
    float xmin_temp = boxes[1];
    float ymax_temp = boxes[2];
    float xmax_temp = boxes[3];

    int xmin = (int) std::max(1.f, xmin_temp * CAMERA_WIDTH);
    int xmax = (int) std::min(CAMERA_WIDTH, xmax_temp * CAMERA_WIDTH);
    int ymin = (int) std::max(1.f, ymin_temp * CAMERA_HEIGHT);
    int ymax = (int) std::min(CAMERA_HEIGHT, ymax_temp * CAMERA_HEIGHT);

    cv::Mat image_freeze;
    cv::cvtColor(frame, image_freeze, cv::COLOR_BGR2RGB);
    cv::resize(image_freeze, image_freeze, cv::Size(width, height), cv::INTER_NEAREST);

    cv::Mat image_result = image_freeze;
    cv::rectangle(image_result, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 1);
    cv::imshow("Output", image_result);
    cv::waitKey(0);

    cv::imwrite("../images/output/identificado/placa_segmentada.png", image_result);

    // Print inference ms in input image
    // cv::putText(frame, "Infernce Time in ms: " + std::to_string(inference_time), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    // ##################### Roi Image ##################### 

    cv::Rect Rec(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    cv::Mat image_roi = image_freeze;
    cv::rectangle(image_roi, Rec, cv::Scalar(0, 255, 0), 1);
    
    cv::Mat roi = image_freeze(Rec);
    cv::resize(roi, roi, cv::Size(200, 31), cv::INTER_NEAREST);
    cv::imshow("Output", roi);      
    cv::waitKey(0);
    cv::imwrite("../images/output/placa/primeiro_completo.png", roi);

    
    // ################## OCR #####################
    // const char *modelFileNameOCR = "../models/ocr/lite-model_keras-ocr_float16.tflite";
    // const char *modelFileNameOCR = "../models/ocr/lite-model_keras-ocr_dr.tflite";
    const char *modelFileNameOCR = "../models/ocr/lite-model_keras-ocr_dr.tflite";
    

    // ##################### Load Model OCR #####################
    std::unique_ptr<tflite::FlatBufferModel> model_ocr = tflite::FlatBufferModel::BuildFromFile(modelFileNameOCR);
    if (model_ocr == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // ##################### Initiate Interpreter #####################
    tflite::ops::builtin::BuiltinOpResolver resolver_ocr;
    std::unique_ptr<tflite::Interpreter> interpreter_ocr;
    tflite::InterpreterBuilder(*model_ocr.get(), resolver_ocr)(&interpreter_ocr);
    if (interpreter_ocr == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    if (interpreter_ocr->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }

    return 0;
}


