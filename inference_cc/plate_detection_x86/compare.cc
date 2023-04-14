#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tesseract/baseapi.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <vector>
#include <dirent.h>
#include <stdio.h>
#include <fstream>



typedef cv::Point3_<float> Pixel;

void normalize(Pixel &pixel){
    pixel.x = ((pixel.x - 255.0) / 255.0);
    pixel.y = ((pixel.y - 255.0) / 255.0);
    pixel.z = ((pixel.z - 255.0) / 255.0);
}

void normalize_integer(Pixel &pixel, float scale, int zero_point){

    pixel.x = ((pixel.x / scale) + zero_point);
    pixel.y = ((pixel.y / scale) + zero_point);
    pixel.z = ((pixel.z / scale) + zero_point);
}


int main(int argc, char **argv)
{

    // FILE *file;
    // file = fopen("/home/pi/plate_detection/result_cc.csv", "w")
    std::ofstream out("/home/pi/plate_detection/result_cc.csv");

    DIR *dir; struct dirent *diread;
    DIR *dir_images; struct dirent *diread_image;
    std::vector<char *> models;
    std::vector<char *> images;
    dir = opendir("../models/plate/");

    while ((diread = readdir(dir)) != nullptr) {

        dir_images = opendir("../images/input");

        if (strcmp(diread->d_name, ".") != 0 && strcmp(diread->d_name, "..") != 0){

            const char* str = diread->d_name;

            std::stringstream ss;
            ss << "../models/plate/" << diread->d_name;
            std::string my_string = ss.str();
            const char *modelFileName = my_string.c_str();

            std::cout << my_string << "\n";

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
            interpreter->SetNumThreads(4);

            // ##################### Get Input Tensor Dimensions #####################
            int input = interpreter->inputs()[0];
            auto height = interpreter->tensor(input)->dims->data[1];
            auto width = interpreter->tensor(input)->dims->data[2];
            auto channels = interpreter->tensor(input)->dims->data[3];

            while ((diread_image = readdir(dir_images)) != nullptr) {

                if (strcmp(diread_image->d_name, ".") != 0 && strcmp(diread_image->d_name, "..") != 0){

                    std::stringstream aa;
                    aa << "../images/input/" << diread_image->d_name;
                    std::string my_string_image = aa.str();
                    const char *imageFile = my_string_image.c_str();

                    std::cout << my_string_image << "\n";

            

                    // ##################### Adjust Image shape (1, 320, 320, 3) ##################### 

                    auto frame = cv::imread(imageFile);
                    cv::Mat image;
                    frame.convertTo(image, CV_32FC3);
                    cv::resize(frame, frame, cv::Size(width, height), cv::INTER_NEAREST);
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

                    Pixel* pixel = image.ptr<Pixel>(0,0);
                    const Pixel* endPixel = pixel + image.cols * image.rows;
                    
                    float* inputImg_ptr = image.ptr<float>(0);
                    float* inputLayer = interpreter->typed_input_tensor<float>(0);
                    int* inputImg_ptr_int = image.ptr<int>(0);
                    int* inputLayer_int = interpreter->typed_input_tensor<int>(0);

                    if (strcmp(diread->d_name, "integer_only.tflite") != 0){
                        for (; pixel != endPixel; pixel++)
                            normalize(*pixel);

                        cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);
                        inputImg_ptr = image.ptr<float>(0);
                        inputLayer = interpreter->typed_input_tensor<float>(0);
                    }
                    else{
                        auto input_tensor = interpreter->inputs()[0];
                        TfLiteTensor *input_class = interpreter->tensor(input_tensor);
                        auto input_quantization = input_class->params;
                        auto input_scale = input_quantization.scale;
                        auto input_zero_point= input_quantization.zero_point;

                        for (; pixel != endPixel; pixel++)
                            normalize_integer(*pixel, input_scale, input_zero_point);

                        cv::resize(image, image, cv::Size(width, height), cv::INTER_NEAREST);
                        image.convertTo(image, CV_8SC3);
                        inputImg_ptr_int = image.ptr<int>(0);
                        inputLayer_int = interpreter->typed_input_tensor<int>(0);
                    }

                    if (strcmp(diread->d_name, "integer_only.tflite") != 0){
                        memcpy(inputLayer, inputImg_ptr, width * height * channels * sizeof(float));
                    }
                    else{
                        memcpy(inputLayer_int, inputImg_ptr_int, width * height * channels * sizeof(int));
                        // memcpy(inputLayer_int, inputImg_ptr_int, image.rows * image.cols * sizeof(int));
                        // memcpy(inputLayer_int, inputImg_ptr_int, image.total() * image.elemSize());
                    }

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

                    // if 'integer_only' in modelo:
                    //     output_scale, output_zero_point = interpreter.get_output_details()[0]['quantization']
                    //     score = (score - output_zero_point) * output_scale

                    // Get count
                    auto count_tensor = interpreter->outputs()[2];
                    TfLiteTensor *output_count = interpreter->tensor(count_tensor);
                    auto output_counts = output_count->data.f;

                    std::cout << diread->d_name << "\n";
                    std::cout << scores[0] << "\n";
                    std::cout << inference_time << "\n";
                    out << diread_image->d_name << "," << diread->d_name << "," << inference_time << "," << scores[0] << "\n";
                }
            }
        }
    }      
    out.close();
    closedir (dir);

    return 0;
}