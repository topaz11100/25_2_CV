#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

template <typename T> static inline T clampT(T v, T lo, T hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

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
    Mat K((int) v.size(), (int) v[0].size(), CV_32F);
    for (int i = 0; i < (int) v.size(); ++i)
        for (int j = 0; j < (int) v[0].size(); ++j)
            K.at<float>(i, j) = v[i][j];
    return K;
}

static void sobel3x3(const Mat& gray, Mat& gx, Mat& gy)
{
    Mat Kx = makeKernel({ {-1,0,1}, {-2,0,2}, {-1,0,1} });
    Mat Ky = makeKernel({ {-1,-2,-1}, {0,0,0}, {1,2,1} });
    gx = conv2D(gray, Kx);
    gy = conv2D(gray, Ky);
}

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
    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            K.at<float>(i, j) = (float) (K.at<float>(i, j) / sum);
    return K;
}

// randn 버전
static Mat addGaussianNoiseU8(const Mat& srcU8, double mean, double stddev, unsigned seed = 7)
{
    CV_Assert(srcU8.type() == CV_8U);
    theRNG().state = seed;               // 시드 고정
    Mat noise(srcU8.size(), CV_32F);
    randn(noise, mean, stddev);          // 정규분포 난수

    Mat srcF; srcU8.convertTo(srcF, CV_32F);
    Mat noisyF = srcF + noise;
    Mat noisyU8;
    noisyF.convertTo(noisyU8, CV_8U);    // 포화 캐스팅
    return noisyU8;
}

// -------------------- Canny 구성요소 --------------------
// 1) Gaussian blur
// 2) Gradient (Sobel), magnitude & angle
// 3) Non-Maximum Suppression
// 4) Double threshold + Hysteresis

static void gradientMagDir(const Mat& gray, Mat& mag, Mat& angleRad)
{
    Mat gx, gy;
    sobel3x3(gray, gx, gy);
    mag.create(gray.size(), CV_32F);
    angleRad.create(gray.size(), CV_32F);
    for (int y = 0; y < gray.rows; ++y)
    {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        float* pm = mag.ptr<float>(y);
        float* pa = angleRad.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x)
        {
            float mx = px[x], my = py[x];
            pm[x] = std::sqrt(mx * mx + my * my);
            pa[x] = std::atan2(my, mx); // -pi ~ pi
        }
    }
}

// 0/45/90/135°로 양자화하여 비최대 억제
static Mat nonMaxSuppression(const Mat& mag, const Mat& angleRad)
{
    Mat out(mag.rows, mag.cols, CV_32F, Scalar(0));
    for (int y = 1; y < mag.rows - 1; ++y)
    {
        const float* pm = mag.ptr<float>(y);
        const float* pa = angleRad.ptr<float>(y);
        float* po = out.ptr<float>(y);
        for (int x = 1; x < mag.cols - 1; ++x)
        {
            float ang = pa[x] * 180.f / (float) CV_PI; // deg
            if (ang < 0) ang += 180.f;
            int dir = 0; // 0,45,90,135
            if ((ang >= 0 && ang < 22.5) || (ang >= 157.5 && ang <= 180)) dir = 0;
            else if (ang >= 22.5 && ang < 67.5) dir = 45;
            else if (ang >= 67.5 && ang < 112.5) dir = 90;
            else dir = 135;

            float v = pm[x];
            float v1 = 0, v2 = 0;
            if (dir == 0) { v1 = mag.at<float>(y, x - 1); v2 = mag.at<float>(y, x + 1); }
            else if (dir == 45) { v1 = mag.at<float>(y - 1, x + 1); v2 = mag.at<float>(y + 1, x - 1); }
            else if (dir == 90) { v1 = mag.at<float>(y - 1, x);   v2 = mag.at<float>(y + 1, x); }
            else { /*135*/      v1 = mag.at<float>(y - 1, x - 1); v2 = mag.at<float>(y + 1, x + 1); }

            po[x] = (v >= v1 && v >= v2) ? v : 0.f;
        }
    }
    return out;
}

static Mat doubleThresholdHysteresis(const Mat& nms, float low, float high)
{
    CV_Assert(nms.type() == CV_32F);
    const uchar STRONG = 255, WEAK = 75, NONE = 0;

    Mat edge(nms.rows, nms.cols, CV_8U, Scalar(0));
    // 1) 이중 임계
    for (int y = 0; y < nms.rows; ++y)
    {
        const float* p = nms.ptr<float>(y);
        uchar* q = edge.ptr<uchar>(y);
        for (int x = 0; x < nms.cols; ++x)
        {
            float v = p[x];
            if (v >= high) q[x] = STRONG;
            else if (v >= low) q[x] = WEAK;
            else q[x] = NONE;
        }
    }
    // 2) 히스테리시스: STRONG과 연결된 WEAK -> STRONG 승격
    auto inside = [&](int r, int c) {
        return (r >= 0 && r < edge.rows && c >= 0 && c < edge.cols);
        };
    std::queue<Point> q;
    vector<vector<bool>> vis(edge.rows, vector<bool>(edge.cols, false));
    for (int y = 0; y < edge.rows; ++y)
    {
        for (int x = 0; x < edge.cols; ++x)
        {
            if (edge.at<uchar>(y, x) == STRONG)
            {
                q.emplace(x, y);
                vis[y][x] = true;
            }
        }
    }
    const int dx[8] = { -1,0,1,-1,1,-1,0,1 };
    const int dy[8] = { -1,-1,-1,0,0,1,1,1 };
    while (!q.empty())
    {
        Point p = q.front(); q.pop();
        for (int k = 0; k < 8; ++k)
        {
            int nx = p.x + dx[k], ny = p.y + dy[k];
            if (!inside(ny, nx)) continue;
            if (!vis[ny][nx] && edge.at<uchar>(ny, nx) == WEAK)
            {
                edge.at<uchar>(ny, nx) = STRONG;
                vis[ny][nx] = true;
                q.emplace(nx, ny);
            }
        }
    }
    // 나머지 WEAK -> NONE
    for (int y = 0; y < edge.rows; ++y)
    {
        for (int x = 0; x < edge.cols; ++x)
        {
            if (edge.at<uchar>(y, x) == WEAK) edge.at<uchar>(y, x) = NONE;
        }
    }
    return edge;
}

static Mat canny_from_scratch(const Mat& grayF, int gksize = 5, double gsigma = 1.0,
    float low = 50.f, float high = 150.f)
{
    // 1) 가우시안 블러
    Mat K = gaussianKernel2D(gksize, gsigma);
    Mat blurF = conv2D(grayF, K);
    // 2) 기울기
    Mat mag, angle;
    gradientMagDir(blurF, mag, angle);
    // 3) NMS
    Mat nms = nonMaxSuppression(mag, angle);
    // 4) Double threshold + Hysteresis
    Mat edges = doubleThresholdHysteresis(nms, low, high);
    return edges;
}

static Mat thresholdByRatio(const Mat& f, float ratio, float maxVal)
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

static Mat sobelAbsSumU8(const Mat& grayF, float ratio = 0.33f, Mat* magOutU8 = nullptr)
{
    float mmax = 0.f;
    // |Gx|+|Gy|
    Mat gx, gy;
    sobel3x3(grayF, gx, gy);
    Mat mag(grayF.rows, grayF.cols, CV_32F);
    for (int y = 0; y < grayF.rows; ++y)
    {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        float* pm = mag.ptr<float>(y);
        for (int x = 0; x < grayF.cols; ++x)
        {
            pm[x] = std::fabs(px[x]) + std::fabs(py[x]);
            if (pm[x] > mmax) mmax = pm[x];
        }
    }
    Mat magU8 = u8FromFloatClamp(mag);
    if (magOutU8) *magOutU8 = magU8.clone();
    Mat bw = thresholdByRatio(mag, ratio, mmax);
    return bw;
}

// -------------------- Program #4 --------------------
// 1) 입력을 그레이(U8)로 만들고 N(0,20) 노이즈 추가
// 2) Canny(직접구현) 수행
// 3) 같은 노이즈 영상에 Sobel |Gx|+|Gy| + max*0.33 임계로 비교
static void program_4(const Mat& bgr)
{
    Mat grayF = toGrayFloat(bgr);
    Mat grayU8 = u8FromFloatClamp(grayF);
    Mat noisy = addGaussianNoiseU8(grayU8, 0.0, 20.0, 2025);
    imshow("p4_noisy.png", noisy);

    Mat noisyF; noisy.convertTo(noisyF, CV_32F);

    // Canny (직접구현) - 파라미터는 필요시 수정
    Mat cannyEdge = canny_from_scratch(noisyF, /*gksize*/5, /*sigma*/1.0, /*low*/50.f, /*high*/150.f);
    imshow("p4_canny.png", cannyEdge);

    // Sobel 비교 (같은 noisy 영상)
    Mat sobelMagU8;
    Mat sobelBW = sobelAbsSumU8(noisyF, 0.33f, &sobelMagU8);
    imshow("p4_sobel_edge.png", sobelMagU8);
    imshow("p4_sobel_thresh.png", sobelBW);

    cerr << "[Program #4] Plotted: noisy, canny, sobel_edge, sobel_thresh\n";
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
    program_4(bgr);
    // plot 유지
    waitKey(0);
    return 0;
}
