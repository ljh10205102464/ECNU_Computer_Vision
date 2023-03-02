#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat src = imread("./magic_mouse.jpg");
    Mat gray = Mat::zeros(src.size(), CV_8UC3);;
 	cvtColor(src, gray, COLOR_BGR2GRAY);//先转为灰度图
    GaussianBlur(gray, gray, Size(3, 3), 3, 3);
 	Mat dst1 = Mat::zeros(src.size(), CV_8UC3);;
 	threshold(gray, dst1, 195, 255, THRESH_BINARY);//二值化阈值处理
    imwrite("dst1.jpg", dst1);
    Mat markImg = Mat::zeros(dst1.size(), CV_8UC3);
    vector<vector<Point>> contours; 
    vector<Vec4i> hierarchy;
    findContours(dst1, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours.size(); i++) {
        cout << "##### contours - " << to_string(i) << "#####" << endl;
        double area = contourArea(contours[i]);
        double len = arcLength(contours[i], true);
        RotatedRect rr = fitEllipse(contours[i]);
        double a = rr.size.width > rr.size.height ? rr.size.width : rr.size.height;
        double c = sqrt(abs(pow(rr.size.width, 2) - pow(rr.size.height, 2)));
        double compactness = len * len / area;
        cout << "compactness:" << compactness << endl;
        double circularity = 4 * CV_PI * area / len / len;
        cout << "circularity:" << circularity << endl;
        double eccentricity = c / a;
        cout << "eccentricity:" << eccentricity << endl;
        drawContours(markImg, contours, i, Scalar(0, 255, 0), 1, 8, hierarchy);
    }
    imwrite("output.png", markImg);
}