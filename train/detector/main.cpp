#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

Net net;

int dnnFaceDetect(Mat &frame, Rect2d &roi, const float confThreshold = 0.5) {
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
//                roi = Rect(left, top, width, height);
            }
            //cout << classId<< " " << confidence<< " " << left<< " " << top<< " " << right<< " " << bottom<< endl;
        }
    }
    return count;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: detector <image_file> [threshold]\n");
        return 1;
    }
    float c = 0.5;
    if (argc > 2) {
        c = atof(argv[2]);
    }

    Mat image = imread(argv[1]);

    Rect2d roi;

    net = dnn::readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");

    int n = dnnFaceDetect(image, roi, c);
    if (n > 0) {
      printf("%d %d %d\n", int(roi.x+roi.width/2), int(roi.y+roi.height/2), int(sqrt(roi.width*roi.height)));
    }

    return 0;
}
