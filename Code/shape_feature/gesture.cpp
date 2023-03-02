#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
using namespace cv::ml;
using namespace std;
using namespace cv;
 
int n_dims = 32;
int n_class = 3;
int n_samples = 30;
// 输入轮廓，得到傅里叶描述子
vector<double> FourierShapeDescriptors(vector<Point>& contour)
{
    vector<double> fd(32);
    //计算轮廓的傅里叶描述子
    Point p;
    int x, y, s;
    int i = 0, j = 0, u = 0;
    s = (int)contour.size();
    float f[9000];//轮廓的实际描述子
    for (u = 0; u < contour.size(); u++)
    {
        float sumx = 0, sumy = 0;
        for (j = 0; j < s; j++)
        {
            p = contour.at(j);
            x = p.x;
            y = p.y;
            sumx += (float)(x * cos(2 * CV_PI * u * j / s) + y * sin(2 * CV_PI * u * j / s));
            sumy += (float)(y * cos(2 * CV_PI * u * j / s) - x * sin(2 * CV_PI * u * j / s));
        }
        f[u] = sqrt((sumx * sumx) + (sumy * sumy));
    }
    //傅立叶描述子的归一化
    f[0] = 0;
    fd[0] = 0;
    for (int k = 2; k < n_dims + 1; k++)
    {
        f[k] = f[k] / f[1];
        fd[k - 1] = f[k];
    }
    return fd;
}
// 输入轮廓，得到Hu矩
vector<double> huDescriptors(vector<Point>& contour) {
    Moments m = moments(contour);
    double hu[7]; vector<double> hus;
    HuMoments(m, hu);
    for (int i = 0; i < 7; i++) {
        hus.push_back(hu[i]);
    }
    return hus;
}
// 基于RGB颜色空间的阈值肤色识别，得到轮廓
vector<Point> findContour(Mat& src) {
    Mat mask = Mat(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            int r, g, b;
            r = src.at<Vec3b>(i, j)[2];
            g = src.at<Vec3b>(i, j)[1];
            b = src.at<Vec3b>(i, j)[0];
 
            if (r > 95 && g > 40 && b > 20 && max(max(r, g), b) - min(min(r, g), b) > 15 && abs(r - g) > 15 && r > g && r > b)
            {
                mask.at<uchar>(i, j) = 255;
            }
            else if (r > 220 && g > 210 && b > 170 && abs(r - g) <= 15 && r > b && g < b)
            {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }  
    Mat ele = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(mask, mask, MORPH_CLOSE, ele);
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    int max_idx = 0; double max_area = 0.0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (max_area < area) {
            max_area = area;
            max_idx = i;
        }
    }
    drawContours(mask, contours, max_idx, Scalar(255), -1);
    imshow("mask", mask);
    waitKey(100);
    return contours[max_idx];
}
// 将描述子保存在本地
void generateTrainData(string& trainImgPath) {
    vector<string> filenames;
    glob(trainImgPath, filenames);
    string s = trainImgPath.substr(trainImgPath.length() - 2, trainImgPath.length() - 1);
    for (int n = 0; n < filenames.size(); n++) {
        Mat src = imread(filenames[n]);
        resize(src, src, Size(1000, 1000));
        vector<Point> contour = findContour(src);
        //vector<double> des = FourierShapeDescriptors(contour, 16);
        //vector<double> des = huDescriptors(contour);
        vector<double> des = FourierShapeDescriptors(contour);
        string txtPath = "./train1/" + s + "_" + to_string(n) + ".txt";
        ofstream file(txtPath);
        for (size_t i = 0; i < des.size(); i++)
        {
            file << des[i] << endl;
        }
        file.close();
    }
}
// 输入图像，获得预测值
int predict(Mat& src)
{
    int dim = n_dims;
    vector<Point> contour = findContour(src);
    //vector<double> des = huDescriptors(contour);
    vector<double> des = FourierShapeDescriptors(contour);
    Ptr<SVM> svm = StatModel::load<SVM>("./svm.xml");
    Mat sample = Mat(1, dim, CV_32FC1);
    float* p = sample.ptr<float>();
    for (int i = 0; i < dim; i++)
    {
        p[i] = des[i];
    }
    Mat result;
    svm->predict(sample, result);
    int pred = int(result.at<float>(0) + 0.5);//四舍五入
    return pred;
}
// 训练svm
void train(string& trainTxtPath) {
    //加载训练数据及标签
    Mat train_data = Mat::zeros(n_samples * n_class, n_dims, CV_32FC1);//每一行是一个训练样本，每一列是特征
    Mat train_label = Mat::zeros(n_samples * n_class, 1, CV_32FC1);//标签
    for (int i = 0; i < n_class; i++)
    {  
        for (int j = 0; j < n_samples; j++)
        {
            int num = i * n_samples + j;
            string filename = "train1/" + to_string(i) + "_" + to_string(j) + ".txt";
            ifstream file(filename);
            int k = 0;
            if (file.is_open()) {
                string line;
                while (getline(file, line)) {
                    double d = stod(line);
                    train_data.at<float>(num, k) = d;
                    k++;
                }
                file.close();
            }
            train_label.at<float>(num, 0) = i;
        }
    }
    //训练
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::NU_SVR);//回归算法
    svm->setKernel(SVM::KernelTypes::RBF);//设置核函数
    svm->setC(8);
    svm->setGamma(1.0/ n_dims);
    svm->setNu(0.5);
    svm->setTermCriteria(TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 500000, 0.0001));
    svm->train(train_data, ROW_SAMPLE, train_label);
    svm->save("./svm.xml");
}
int main() {
    // 生成数据
    for (int n = 0; n < n_class; n++) {
        string trainImgPath = "./石头剪刀布/"+to_string(n);
        generateTrainData(trainImgPath);
    }
    // 训练
    string trainTxtPath = "./";
    train(trainTxtPath);
    vector<string> filenames;
    glob("./石头剪刀布/test", filenames);
    // 测试
    for (int n = 0; n < filenames.size(); n++) {
        string filename = filenames[n];
        int idx = filename.find_last_of("-");
        string c = filename.substr(idx-1, idx);
        Mat src = imread(filename);
        resize(src, src, Size(1000, 1000));
        int pred = predict(src);
        cout << "label:" << c[0] << ", predict:" << pred << endl;
    }
 
    return 0;
}