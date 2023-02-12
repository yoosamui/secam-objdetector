//  gcc main.cpp `pkg-config opencv4 --libs --cflags` -o main -lstdc++ -pthread
//  g++ -O3 main.cpp -o main `pkg-config --cflags --libs opencv4` +++
//
//  Thanks and all credits to:
//  Luiz doleron
//  https://medium.com/mlearning-ai/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
//
//  https://morioh.com/p/a7dae7ba49d2
//
//  git clone https://github.com/ultralytics/yolov5  # clone
//  cd yolov5
//  pip install -r requirements.txt  # install
//
//  I get this code from here:
//  https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/cpp/yolo.cpp
//  https://github.com/nandinib1999/object-detection-yolo-opencv
//
//  https://ultralytics.com/yolov5
//
#include <dirent.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

std::vector<std::string> classes;

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("data/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}
void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("data/yolov5n.onnx");
    if (is_cuda) {
        //    std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        //   std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
                                        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.65;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};
// clang-format off
    map<string, Scalar> color_map = {
        {"person", Scalar(0, 255, 255)},
        {"motorbike", Scalar(255, 255, 0)},
        {"car", Scalar(0, 255, 0)}
    };
// clang-format on
Scalar get_color(const string &name)
{
    if (color_map.count(name) == 0) return Scalar(255, 255, 255);

    return color_map[name];
}

cv::Mat format_yolov5(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(float confidence_threshold, cv::Mat &image, cv::dnn::Net &net,
            std::vector<Detection> &output, const std::vector<std::string> &className)
{
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                           cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= confidence_threshold) {
            float *classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}
Rect inflate(const Rect &rect, size_t size, const Mat &frame)
{
    Rect r = rect;

    r.x -= size;
    if (r.x < 0) r.x = 0;

    r.y -= size;
    if (r.y < 0) r.y = 0;

    r.width += size * 2;
    if (r.x + r.width > frame.cols) {
        r.x -= (r.x + r.width) - frame.cols;
    }

    r.height += size * 2;
    if (r.y + r.height > frame.rows) {
        r.y -= (r.y + r.height) - frame.rows;
    }

    return r;
}
bool draw(const Mat &frame, vector<Detection> &output, const string &output_file)
{
    int detections = output.size();
    if (!detections) return false;

    Mat input;
    frame.copyTo(input);

    std::ifstream infile("includeonly.txt");
    std::string line;
    string keyname;

    vector<string> includeonly;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        if (line[0] == '*') {
            keyname = line.substr(1);
            line = keyname;
        }

        includeonly.push_back(line);
    }

    bool found = false;
    //    cout << "draw : " << detections << endl;
    for (int c = 0; c < detections; ++c) {
        auto detection = output[c];
        auto box = detection.box;
        auto classId = detection.class_id;
        float confidence = detection.confidence;
        auto color = get_color(classes[classId]);

        if (includeonly.size()) {
            string classname = classes[classId];

            std::vector<string>::iterator it =
                std::find(includeonly.begin(), includeonly.end(), classname);

            if (it == includeonly.end()) {
                continue;
            }

            if (!keyname.empty()) {
                found = keyname == classname;
            } else {
                found = true;
            }
        } else {
            found = true;
        }

        //   cout << "found classname : " << classes[classId] << endl;

        Rect r = inflate(box, 20, input);

        rectangle(input, r, color, 2);
        rectangle(input, Point(r.x - 1, r.y - 20), cv::Point(r.x + r.width, r.y), color, FILLED);

        float fscale = 0.4;
        int thickness = 1;
        string title = classes[classId] + " " + to_string(confidence);

        //            cout << c << " " << r.x << " " << title << endl;
        putText(input, title, Point(r.x + 2, r.y - 5), cv::FONT_HERSHEY_SIMPLEX, fscale,
                Scalar(0, 0, 0), thickness, LINE_AA, false);
    }

    if (found) imwrite(output_file, input);
    return found;
}
bool SortComparator(string s1, string s2)
{
    return s1 < s2;
}
// static const string keys =
//"{ help help    |    | print help message. }"
//"{ confidence c |    | the detection confidence 0.0 - 1.0 "
//"{ directory d  |    | the image directory.}"
//"{ targetfile t |    | destination file.}";
static const string keys =
    "{ help help|       | print help message. }"
    "{ confidence c |0.4| the detection confidence 0.0 - 1.}"
    "{ directory d  |   | the image directory.}"
    "{ targetfile t |output.jpg  | destination file.}";

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("This is the secam client server system.");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    float confidence = parser.get<float>("confidence");
    string directory = parser.get<std::string>("directory");
    string targetfile = parser.get<std::string>("targetfile");

    // cout << "---CONFIDENCE = " << confidence << endl;
    // cout << "---DIRECTORY = " << directory << endl;
    // cout << "---TARGETFILE = " << targetfile << endl;

    std::ifstream ifs("data/classes.txt");
    classes = load_class_list();

    // Read all images form Directory.
    DIR *dir;
    struct dirent *ent;
    vector<string> images;

    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0) continue;
            if (strcmp(ent->d_name, "..") == 0) continue;

            images.push_back(ent->d_name);
        }
        closedir(dir);
    } else {
        // could not open directory
        perror("");
        return EXIT_FAILURE;
    }

    bool is_cuda = false;  // argc > 1 && strcmp(argv[1], "cuda") == 0;
    cv::dnn::Net net;

    bool found = false;
    if (images.size()) {
        // sort(images.begin(), images.end(), greater<string>());
        sort(images.begin(), images.end(), SortComparator);  // this will sort the strings

        // for (std::vector<std::string>::iterator it = images.begin(); it != images.end(); ++it) {
        // std::cout << *it << std::endl;
        //}
        for (auto &s : images) {
            string filename = directory + "/" + s;

            //  std::cout << filename << std::endl;
            Mat frame = imread(filename);

            load_net(net, is_cuda);

            //      cout << "DETECT-----> " << filename << endl;
            std::vector<Detection> output;
            detect(confidence, frame, net, output, classes);

            int detections = output.size();
            //    cout << "detections: " << detections << endl;
            if (draw(frame, output, targetfile)) {
                found = true;
                //      cout << "Found!" << endl;
                break;
            }
        }
    }

    cout << (int)found << endl;

    return (int)found;
}
