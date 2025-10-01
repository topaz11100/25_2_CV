#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void programming_3(Mat& src, Mat& out)
{
    out.create(src.size(), CV_8UC1);
    for (int y = 0; y < out.rows; y++)
    {
        for (int x = 0; x < out.cols; x++)
        {
            Vec3b &pixel = src.at<Vec3b>(y, x);
            double B = static_cast<double>(pixel[0]);
            double G = static_cast<double>(pixel[1]);
            double R = static_cast<double>(pixel[2]);

            uchar Y = static_cast<uchar>(0.299 * R + 0.587 * G + 0.114 * B);
            out.at<uchar>(y, x) = Y;
        }
    }
}

void programming_4(Mat &src, Mat &out)
{
    // 출력도 3채널 (YUV 3채널)
    out.create(src.size(), CV_8UC3);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            Vec3b pixel = src.at<Vec3b>(y, x);

            double B = static_cast<double>(pixel[0]);
            double G = static_cast<double>(pixel[1]);
            double R = static_cast<double>(pixel[2]);

            // YUV 변환
            double Y = 0.299 * R + 0.587 * G + 0.114 * B;
            double U = -0.147 * R - 0.289 * G + 0.436 * B + 128;
            double V = 0.615 * R - 0.515 * G - 0.100 * B + 128;

            // 범위 클리핑 (0~255)
            int Yi = static_cast<int>(Y);
            int Ui = static_cast<int>(U);
            int Vi = static_cast<int>(V);

            if (Yi < 0)
                Yi = 0;
            if (Yi > 255)
                Yi = 255;
            if (Ui < 0)
                Ui = 0;
            if (Ui > 255)
                Ui = 255;
            if (Vi < 0)
                Vi = 0;
            if (Vi > 255)
                Vi = 255;

            // out 채널 저장 (YUV 순서로)
            out.at<Vec3b>(y, x)[0] = static_cast<uchar>(Yi); // Y
            out.at<Vec3b>(y, x)[1] = static_cast<uchar>(Ui); // U
            out.at<Vec3b>(y, x)[2] = static_cast<uchar>(Vi); // V
        }
    }
}

void saveYUV444(const Mat &yuv, const string &filename)
{
    int width = yuv.cols;
    int height = yuv.rows;

    ofstream fout(filename, ios::out | ios::binary);
    if (!fout.is_open())
    {
        cerr << "파일 열기 실패: " << filename << endl;
        return;
    }

    // Y, U, V 채널 분리 후 저장
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fout.put(yuv.at<Vec3b>(y, x)[0]); // Y
        }
    }
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fout.put(yuv.at<Vec3b>(y, x)[1]); // U
        }
    }
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fout.put(yuv.at<Vec3b>(y, x)[2]); // V
        }
    }

    fout.close();
}

int main()
{
    Mat lena = imread("Lenna.png", IMREAD_COLOR);
    imshow("source", lena);

    Mat gray;
    programming_3(lena, gray);
    imshow("gray_converted", gray);

    Mat yuv;
    programming_4(lena, yuv); // BGR → YUV444 변환
    saveYUV444(yuv, "yuv_converted.yuv");

    waitKey(0);

    return 0;
}