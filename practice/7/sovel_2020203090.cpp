#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <string>
#include <limits>

using namespace cv;
using namespace std;

// -------------------- 유틸 --------------------
template <typename T> static inline T clampT(T v, T lo, T hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// BGR -> GRAY (float, 0~255)
// OpenCV의 cvtColor 미사용: 직접 계산 (BT.601 근사)
static Mat toGrayFloat(const Mat& bgr)
{
    CV_Assert(bgr.channels() == 3);
    Mat gray(bgr.rows, bgr.cols, CV_32F);
    for (int y = 0; y < bgr.rows; ++y)
    {
        const Vec3b* row = bgr.ptr<Vec3b>(y);
        float* gout = gray.ptr<float>(y);
        for (int x = 0; x < bgr.cols; ++x)
        {
            // BGR 순서
            float B = (float) row[x][0];
            float G = (float) row[x][1];
            float R = (float) row[x][2];
            gout[x] = 0.114f * B + 0.587f * G + 0.299f * R;
        }
    }
    return gray;
}

static Mat u8FromFloatClamp(const Mat& f)
{
    Mat out(f.rows, f.cols, CV_8U);
    for (int y = 0; y < f.rows; ++y)
    {
        const float* p = f.ptr<float>(y);
        uchar* q = out.ptr<uchar>(y);
        for (int x = 0; x < f.cols; ++x)
        {
            int v = (int) std::round(p[x]);
            q[x] = (uchar) clampT<int>(v, 0, 255);
        }
    }
    return out;
}

// 경계는 replicate로 처리하며 float 컨볼루션
static Mat conv2D(const Mat& src, const Mat& kernel)
{
    CV_Assert(src.type() == CV_32F && kernel.type() == CV_32F);
    CV_Assert(kernel.rows % 2 == 1 && kernel.cols % 2 == 1);
    int kr = kernel.rows / 2, kc = kernel.cols / 2;
    Mat dst(src.rows, src.cols, CV_32F, Scalar(0));
    for (int y = 0; y < src.rows; ++y)
    {
        float* drow = dst.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            double acc = 0.0;
            for (int ky = -kr; ky <= kr; ++ky)
            {
                int sy = clampT<int>(y + ky, 0, src.rows - 1);
                const float* srow = src.ptr<float>(sy);
                const float* krow = kernel.ptr<float>(ky + kr);
                for (int kx = -kc; kx <= kc; ++kx)
                {
                    int sx = clampT<int>(x + kx, 0, src.cols - 1);
                    acc += srow[sx] * krow[kx + kc];
                }
            }
            drow[x] = (float) acc;
        }
    }
    return dst;
}

static Mat makeKernel(const vector<vector<float>>& v)
{
    int r = (int) v.size(), c = (int) v[0].size();
    Mat K(r, c, CV_32F);
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j) K.at<float>(i, j) = v[i][j];
    }
    return K;
}

static void sobel3x3(const Mat& gray, Mat& gx, Mat& gy)
{
    // 표준 Sobel 3x3
    Mat Kx = makeKernel({ {-1,0,1}, {-2,0,2}, {-1,0,1} });
    Mat Ky = makeKernel({ {-1,-2,-1}, {0,0,0}, {1,2,1} });
    gx = conv2D(gray, Kx);
    gy = conv2D(gray, Ky);
}

// |Gx| + |Gy| (슬라이드 요구)
static Mat sobelAbsSum(const Mat& gray, float& maxValOut)
{
    Mat gx, gy;
    sobel3x3(gray, gx, gy);
    Mat mag(gray.rows, gray.cols, CV_32F);
    float mmax = 0.f;
    for (int y = 0; y < gray.rows; ++y)
    {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        float* pm = mag.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x)
        {
            float v = std::fabs(px[x]) + std::fabs(py[x]);
            pm[x] = v;
            if (v > mmax) mmax = v;
        }
    }
    maxValOut = mmax;
    return mag;
}

static Mat thresholdByRatio(const Mat& f, float ratio /*e.g., 0.33*/, float maxVal)
{
    Mat out(f.rows, f.cols, CV_8U);
    float th = ratio * maxVal;
    for (int y = 0; y < f.rows; ++y)
    {
        const float* p = f.ptr<float>(y);
        uchar* q = out.ptr<uchar>(y);
        for (int x = 0; x < f.cols; ++x) q[x] = (p[x] >= th ? 255 : 0);
    }
    return out;
}

// 가우시안 커널 생성 (2D), 합=1
static Mat gaussianKernel2D(int ksize, double sigma)
{
    CV_Assert(ksize % 2 == 1 && sigma > 0.0);
    int r = ksize / 2;
    Mat K(ksize, ksize, CV_32F);
    double s2 = 2 * sigma * sigma;
    double sum = 0.0;
    for (int y = -r; y <= r; ++y)
    {
        for (int x = -r; x <= r; ++x)
        {
            double v = std::exp(-(x * x + y * y) / s2);
            K.at<float>(y + r, x + r) = (float) v;
            sum += v;
        }
    }
    // 정규화
    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            K.at<float>(i, j) = (float) (K.at<float>(i, j) / sum);
    return K;
}

// 가우시안 노이즈 추가: N(mean, std) — randn 버전
static Mat addGaussianNoiseU8(const Mat& srcU8, double mean, double stddev, unsigned seed = 42)
{
    CV_Assert(srcU8.type() == CV_8U);
    theRNG().state = seed;                    // 재현성 보장
    Mat noise(srcU8.size(), CV_32F);
    randn(noise, mean, stddev);               // 평균 mean, 표준편차 stddev의 정규분포 난수

    Mat srcF;  srcU8.convertTo(srcF, CV_32F); // 원본을 float로
    Mat noisyF = srcF + noise;                // 더하기
    Mat noisyU8;
    noisyF.convertTo(noisyU8, CV_8U);         // 포화 캐스팅으로 0~255 클리핑
    return noisyU8;
}

// -------------------- Program #1 --------------------
// 1) BGR 입력 -> GRAY(float)
// 2) Sobel |Gx|+|Gy|
// 3) threshold = max * 0.33
static void program_1(const Mat& bgr, const string& tag = "p1")
{
    Mat grayF = toGrayFloat(bgr);
    float mmax = 0.f;
    Mat mag = sobelAbsSum(grayF, mmax);
    Mat magU8 = u8FromFloatClamp(mag);
    Mat bw = thresholdByRatio(mag, 0.33f, mmax);
    imshow(tag + "_edge.png", magU8);
    imshow(tag + "_thresh.png", bw);
    cerr << "[Program #1] max(|Gx|+|Gy|)=" << mmax << ", threshold=" << (0.33f * mmax) << endl;
}

// -------------------- Program #2 --------------------
// 1) 입력 그레이 변환 후 노이즈 N(0,20) 추가
// 2) Program #1과 동일한 방식으로 에지 검출
static void program_2(const Mat& bgr)
{
    Mat grayF = toGrayFloat(bgr);
    Mat grayU8 = u8FromFloatClamp(grayF);
    Mat noisy = addGaussianNoiseU8(grayU8, 0.0, 20.0, 2025);
    imshow("p2_noisy.png", noisy);

    Mat noisyF; noisy.convertTo(noisyF, CV_32F);
    float mmax = 0.f;
    Mat mag = sobelAbsSum(noisyF, mmax);
    Mat magU8 = u8FromFloatClamp(mag);
    Mat bw = thresholdByRatio(mag, 0.33f, mmax);

    imshow("p2_edge.png", magU8);
    imshow("p2_thresh.png", bw);
    cerr << "[Program #2] noisy max(|Gx|+|Gy|)=" << mmax << ", threshold=" << (0.33f * mmax) << endl;
}

// -------------------- Program #3 --------------------
// 1) 입력 그레이 변환
// 2) 가우시안 블러(직접 구현) -> (ksize=5, sigma=1.0 예시)
// 3) Program #1과 동일한 Sobel + 이진화
static void program_3(const Mat& bgr)
{
    Mat grayF = toGrayFloat(bgr);
    Mat K = gaussianKernel2D(5, 1.0);
    Mat blurF = conv2D(grayF, K);
    Mat blurU8 = u8FromFloatClamp(blurF);
    imshow("p3_blur.png", blurU8);

    float mmax = 0.f;
    Mat mag = sobelAbsSum(blurF, mmax);
    Mat magU8 = u8FromFloatClamp(mag);
    Mat bw = thresholdByRatio(mag, 0.33f, mmax);
    imshow("p3_edge.png", magU8);
    imshow("p3_thresh.png", bw);
    cerr << "[Program #3] blurred max(|Gx|+|Gy|)=" << mmax << ", threshold=" << (0.33f * mmax) << endl;
}

int main()
{
    string inPath = "/home/yongokhan/바탕화면/25_2_CV/source/lena.png";
    Mat bgr = imread(inPath, IMREAD_COLOR);
    if (bgr.empty())
    {
        cerr << "Failed to read image: " << inPath << "\n";
        return 1;
    }

    // Program #1
    program_1(bgr, "p1");

    // Program #2
    program_2(bgr);

    // Program #3
    program_3(bgr);

    // plot 유지
    waitKey(0);
    return 0;
}
