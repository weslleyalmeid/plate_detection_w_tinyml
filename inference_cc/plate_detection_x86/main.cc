#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tesseract/baseapi.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"


typedef cv::Point3_<float> Pixel;

void normalize(Pixel &pixel){
    pixel.x = ((pixel.x - 255.0) / 255.0);
    pixel.y = ((pixel.y - 255.0) / 255.0);
    pixel.z = ((pixel.z - 255.0) / 255.0);
}


int main(int argc, char **argv)
{

    // const char *modelFileName = argv[1];
    // const char *labelFile = argv[2];
    // const char *imageFile = argv[3];

    const char *modelFileName = "../models/plate/float16.tflite";
    // const char *modelFileName = "../models/classificao/mobilenet_v1_1.0_224_quant.tflite";
    // const char *imageFile = "../images/input/BAG-7751.jpg";
    // const char *imageFile = "../images/input/EUL-0433.jpg";
    // const char *imageFile = "../images/input/FGH-2302.jpg";
    
    // const char *imageFile = "../images/input/Cars401.png";
    // const char *imageFile = "../images/input/Cars417.png";

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
    float threshold = 0.7f;
    
    // ##################### Get Input Tensor Dimensions #####################
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];

    // ################## OCR #####################
    double scale_up_x = 6;
    double scale_up_y = 6;
    int low = 70;
    int up = 255;

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    // api->Init(NULL, "eng", tesseract::OEM_TESSERACT_ONLY);

    // api->SetPageSegMode(tesseract::PSM_AUTO);
    api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    api->SetVariable("tessedit_char_whitelist","ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");


     // ################## ADJUST CAMERA #####################

    auto cam = cv::VideoCapture(0);
    // auto cam = cv::VideoCapture("../images/input/BAG-7751.jpg");
    // auto cam = cv::VideoCapture("../images/placas.mp4");

    int fps = cam.get(cv::CAP_PROP_FPS);
    // cam.set(cv::CAP_PROP_FPS, 25);
    std::cout << "FPS: " << fps << std::endl;

    int num_frames = 1;

    int frameDuration = 1000 / fps;  // frame duration in milliseconds
    std::cout << "frame duration: " << frameDuration << " ms" << std::endl;
    auto cam_width =cam.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::Mat frame;
    
    clock_t startfps;
    clock_t endfps;

    while (true) {

        startfps = clock();

        auto success = cam.read(frame);
        if (!success) {
            std::cout << "cam fail" << std::endl;
            break;
        }
        // ##################### Adjust Image shape (1, 320, 320, 3) ##################### 
        // cv::imshow("cam", frame);
        
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

        std::cout << "Inference time:" + inference_time;

        std::vector<std::pair<float, int>> result;

        // Get scores
        auto scores_tensor = interpreter->outputs()[0];
        TfLiteTensor *output_score = interpreter->tensor(scores_tensor);
        auto output_scores = output_score->data.f;

        if(output_scores[0] > threshold){

            auto box_tensor = interpreter->outputs()[1];
            TfLiteTensor *output_box = interpreter->tensor(box_tensor);
            auto output_boxes = output_box->data.f;
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

            // Get count
            auto count_tensor = interpreter->outputs()[2];
            TfLiteTensor *output_count = interpreter->tensor(count_tensor);
            auto output_counts = output_count->data.f;

        
        // ##################### Image Detection ##################### 
        
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
            cv::resize(image_result, image_result, cv::Size(frame.cols, frame.rows));
            cv::imshow("cam", image_result);
            cv::waitKey(1);

            // ##################### Roi Image ##################### 

            cv::Rect Rec(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
            cv::Mat image_roi = image_freeze;
            cv::rectangle(image_roi, Rec, cv::Scalar(0, 255, 0), 1);
            
            cv::Mat roi = image_freeze(Rec);

            // ################## OCR #####################
            cv::Mat roi_tesseract;
            cv::Mat image_final;
            cv::resize(roi, image_final, cv::Size(), scale_up_x, scale_up_y, cv::INTER_CUBIC);
            cv::cvtColor(image_final, roi_tesseract, cv::COLOR_BGR2GRAY);

            cv::threshold(roi_tesseract, roi_tesseract, low, up, cv::THRESH_BINARY);


            api->SetVariable("user_defined_dpi", "70");
            api->SetImage(roi.data, roi.cols, roi.rows, 3, roi.step);
            std::string outText = std::string(api->GetUTF8Text());
            std::cout << outText;

            cv::putText(image_result, outText, {50,50}, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("cam", image_result);
            cv::waitKey(1);

        }

        else{
            endfps = clock();
            double seconds =  ((double) endfps - (double) startfps) / (double) CLOCKS_PER_SEC;

            double fpsLive = (double)num_frames / (double)seconds;

            cv::putText(frame, "FPS: " + std::to_string(fpsLive), {50,50}, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
            cv::imshow("cam", frame);
            cv::waitKey(1);

        }

    }

    return 0;
}