#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

using namespace cv;
using namespace std;


int main()
{
    Mat lena_color = imread("/home/yongokhan/바탕화면/25_2_CV/source/lena.png", IMREAD_COLOR);
    Mat lena_gray  = imread("/home/yongokhan/바탕화면/25_2_CV/source/lena.png", IMREAD_GRAYSCALE);
    
    imshow("color", lena_color);
    imshow("gray", lena_gray);

    waitKey(0);

    return 0;
}