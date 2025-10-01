#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 영상 좌우 반전
void flipHorizontal(Mat &src, Mat &dst)
{
    dst.create(src.size(), src.type()); 
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            dst.at<uchar>(y, x) = src.at<uchar>(y, src.cols - 1 - x);
        }
    }
}

// 영상 상하 반전
void flipVertical(Mat &src, Mat &dst)
{
    dst.create(src.size(), src.type()); 
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            dst.at<uchar>(y, x) = src.at<uchar>(src.rows - 1 - y, x);
        }
    }
}

// 영상 전체에 특정 값을 더하거나 빼는 함수
void addValue(Mat &src, Mat &dst, int value)
{
    dst.create(src.size(), src.type()); 
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            int pixel = src.at<uchar>(y, x) + value;
            if (pixel < 0)
                pixel = 0;
            if (pixel > 255)
                pixel = 255;
            dst.at<uchar>(y, x) = static_cast<uchar>(pixel);
        }
    }
}

// 두 영상의 평균
void averageImage(Mat &src1, Mat &src2, Mat &dst)
{
    dst.create(src1.size(), src1.type());
    for (int y = 0; y < src1.rows; y++)
    {
        for (int x = 0; x < src1.cols; x++)
        {
            dst.at<uchar>(y, x) = (src1.at<uchar>(y, x) + src2.at<uchar>(y, x)) / 2;
        }
    }
}

// 영상 A - B
void subtractImage(Mat &src1, Mat &src2, Mat &dst)
{
    dst.create(src1.size(), src1.type());
    for (int y = 0; y < src1.rows; y++)
    {
        for (int x = 0; x < src1.cols; x++)
        {
            int pixel = src1.at<uchar>(y, x) - src2.at<uchar>(y, x);
            if (pixel < 0)
                pixel = 0;
            dst.at<uchar>(y, x) = static_cast<uchar>(pixel);
        }
    }
}

int main()
{
    // lena 영상 흑백으로 load
    Mat lena = imread("Lenna.png", IMREAD_GRAYSCALE);

    Mat flipH, flipV, addImg, avgImg, subImg;

    // 반전
    flipHorizontal(lena, flipH);
    flipVertical(lena, flipV);

    // 더하기
    addValue(lena, addImg, 50);

    // 평균 (여기선 예시로 lena와 flipH의 평균)
    averageImage(lena, flipH, avgImg);

    // 차연산 (lena - flipH)
    subtractImage(lena, flipH, subImg);

    imshow("Original", lena);
    imshow("Flip Horizontal", flipH);
    imshow("Flip Vertical", flipV);
    imshow("Add +50", addImg);
    imshow("Average", avgImg);
    imshow("Subtract", subImg);

    waitKey(0);
    return 0;
}
