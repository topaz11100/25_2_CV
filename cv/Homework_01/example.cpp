#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat img_in;

	img_in = imread("Lena.png");
	cvtColor(img_in, img_in, cv::COLOR_RGB2GRAY);
	imshow("source img", img_in);

	unsigned char* pData = (unsigned char *)img_in.data;

	for (int i = 0; i < img_in.cols*img_in.rows; i++)
	{
		pData[i] = 0;
	}
	imshow("output image", img_in);

	waitKey(0);

	return 0;
}