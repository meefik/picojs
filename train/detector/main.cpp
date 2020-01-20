#include <dirent.h>
#include <string>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int dnnFaceDetect(Net &net, Mat &frame, Rect2d &roi, const float confThreshold = 0.5) {
    Mat blob = dnn::blobFromImage(frame, 1, Size(300,300), Scalar(0,0,0,0), false, CV_8U);
    net.setInput(blob);
    Mat res = net.forward("detection_out");
    Mat faces(res.size[2],res.size[3], CV_32F, res.ptr<float>());
    int max_area = 0;
    int count = 0;
    for (int i=0; i<faces.rows; i++)
    {
        float *data = faces.ptr<float>(i);
        float confidence = data[2];
        if (confidence > confThreshold)
        {
            count++;
            int left = (int)(data[3] * frame.cols);
            int top = (int)(data[4] * frame.rows);
            int right = (int)(data[5] * frame.cols);
            int bottom = (int)(data[6] * frame.rows);
            int classId = (int)(data[1]) - 1;  // Skip 0th background class id.
            left = min(max(0, left), frame.cols - 1);
            right = min(max(0, right), frame.cols - 1);
            bottom = min(max(0, bottom), frame.rows - 1);
            top = min(max(0, top), frame.rows - 1);
            int width = right - left;
            int height = bottom - top;
            int area = width * height;
            if (left < right && top < bottom && area > max_area) {
                max_area = area;
                roi.x = left;
                roi.y = top;
                roi.width = width;
                roi.height = height;
            }
        }
    }
    return count;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: detector <path> [threshold]" << std::endl;
        return 1;
    }
    char *path = argv[1];
    float threshold = 0.87;
    if (argc > 2) {
        threshold = atof(argv[2]);
    }

    Net net = dnn::readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");
    Mat image;
    Rect2d roi;

    DIR *dir;
    dirent *pdir;
    int n;
    dir = opendir(path);
    while((pdir = readdir(dir)) != NULL)
    {
        if (pdir->d_type != DT_REG) continue;
        string fp = string(path) + pdir->d_name;
        image = imread(fp);
        n = dnnFaceDetect(net, image, roi, threshold);
        if (n == 1) {
            std::cout << int(roi.x+roi.width/2);
            std::cout << "\t";
            std::cout << int(roi.y+roi.height/2);
            std::cout << "\t";
            std::cout << int(sqrt(roi.width*roi.height));
            std::cout << "\t";
            std::cout << fp.c_str() << std::endl;
        } else if (n == 0) {
            std::cerr << "0\t0\t0\t";
            std::cerr << fp.c_str() << std::endl;
        }
    }
    closedir(dir);

    return 0;
}
