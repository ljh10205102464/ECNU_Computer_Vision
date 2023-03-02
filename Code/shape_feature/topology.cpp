#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat src = imread("./4.bmp", 0);
    Mat markImg = Mat::zeros(src.size(), CV_8UC3);
    vector<vector<Point>> contours; vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {
            int c = 1; int h = 0;
            drawContours(markImg, contours, i, Scalar(0, 255, 0), 1);
            int child_index = hierarchy[i][2];
            while (child_index != -1) {
                h++;
                drawContours(markImg, contours, child_index, Scalar(0, 0, 255), 1);
                int next_index = hierarchy[child_index][0];
                child_index = next_index;
            }
            cout << "euler:" << c - h << endl;
        }
    }
    imwrite("output_topology.png", markImg);
}