#include "model/UltraFace.h"
#include <iostream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <chrono>
#include <thread>

struct data
{
    float embed[128];
    int label;
};

std::vector<data> load_database(const std::string &path)
{
    std::vector<data> database;
    data person;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        for (int i = 0; i < 128; i++)
        {
            iss >> person.embed[i];
        }
        iss >> person.label;

        database.push_back(person);
    }
    return database;
}
float cosin(const ncnn::Mat &tensor, const float *arr)
{
    float dot_product = 0.0;
    for (int i = 0; i < 128; i++)
    {
        dot_product += tensor.channel(0)[i] * arr[i];
    }

    float norm_a = 0.0;
    float norm_b = 0.0;
    for (int i = 0; i < 128; i++)
    {
        norm_a += tensor.channel(0)[i] * tensor.channel(0)[i];
        norm_b += arr[i] * arr[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    // Tính cosine similarity
    float cosine_similarity = dot_product / (norm_a * norm_b);
    return cosine_similarity;
}
ncnn::Mat pre_processing(const cv::Mat &image)
{
    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};
    ncnn::Mat img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    ncnn::Mat in;
    ncnn::resize_bilinear(img, in, 96, 112);
    ncnn::Mat ncnn_img = in;
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
    return ncnn_img;
}

std::tuple<int, float> post_processing(const ncnn::Mat &tensor, const std::string &path)
{
    std::vector<data> database = load_database(path);
    int index = -1;
    float max, cos;
    for (int i = 0; i < database.size(); i++)
    {
        cos = cosin(tensor, database[i].embed);
        if (max < cos)
        {
            max = cos;
            index = i;
        }
    }
    if (max < 0.5)
    {
        return std::make_tuple(1, max);
    }
    return std::make_tuple(database[index].label, max);
}

void save_tensor_txt(const ncnn::Mat &mat, const std::string &filename, const int label)
{
    std::ofstream file(filename, std::ios::app);
    for (int i = 0; i < mat.w; ++i)
    {
        float value = mat.channel(0)[i];
        file << value << " ";
    }
    file << label << "\n";
}
std::vector<std::string> get_name(const std::string &filename)
{
    std::vector<std::string> lines;
    std::ifstream file(filename);
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            lines.push_back(line);
        }
    }
    file.close();
    return lines;
}
int main()
{   

    std::string bin_path_detect = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/weight/slim_320.bin";
    std::string param_path_detect = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/weight/slim_320.param";

    std::string bin_path_recog = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/weight/custom.bin";
    std::string param_path_recog = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/weight/custom.param";
    UltraFace ultraface(bin_path_detect, param_path_detect, 320, 240, 1, 0.7);

    std::string file_data = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/data/database.txt";
    std::string file_name = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/data/name.txt";

    std::vector<std::string> list_name = get_name(file_name);
    ncnn::Net model;
    if (model.load_param(param_path_recog.data()))
        exit(-1);
    if (model.load_model(bin_path_recog.data()))
        exit(-1);
    ncnn::Extractor ex = model.create_extractor();
    std::string rtsp = "rtsp://10.2.204.114:554/live0";
    // std::string rtsp = "rtsp://192.168.88.170:554/live0";
    // std::string video_path = "/home/congnt/congnt/VNPT/Face_Recognition_ncnn/videos/data/taoquan.mp4";
    // cv::VideoCapture cap(video_path);
    cv::VideoCapture cap(rtsp);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera!" << std::endl;
        return -1;
    }

    int key,i;
    ncnn::Mat output_recog;
    ncnn::Mat input_recog;
    while (true)
    {
        cv::Mat frame;
        cap.read(frame);
        key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);
        if (face_info.size() > 0)
        {
            for (i = 0; i < face_info.size(); i++)
            {   
                auto face = face_info[i];
                cv::Point pt1(face.x1, face.y1);
                cv::Point pt2(face.x2, face.y2);
                cv::Rect roi(pt1, pt2);
                cv::Mat cropped_image = frame(roi).clone();
                input_recog = pre_processing(cropped_image);
                ncnn::Extractor ex = model.create_extractor();
                ex.input("input", input_recog);
                ex.extract("output", output_recog);
                
                // if index = 0 , not person in data
                std::tuple<int, float> result = post_processing(output_recog, file_data);
                int id = std::get<0>(result);
                float cos_value = std::get<1>(result);
                std::cout<<cos_value<<std::endl;
                cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                // std::string name = list_name[id] + std::to_string(cos_value);
                std::string name = list_name[id-1];
                cv::putText(frame, name, cv::Point(pt1.x, pt2.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }
        }
        cv::imshow("Camera", frame);
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
