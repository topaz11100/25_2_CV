// 실행 방법: cli로 인수를 넣어 실행
//   > 3.exe path/input.png 3.0 160 0.0 0.0  
//   파일 경로, 보간 스케일, 매칭할 확률분포의 pmf(0), pmf(128), pmf(255) 값
// ------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// ---------- 공통 유틸 ----------

// 0~255 범위로 자르기
static inline uint8_t sat_u8(int v)
{
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return static_cast<uint8_t>(v);
}

// 인덱스를 [0, max-1]에 고정(경계 복제: BORDER_REPLICATE)
static inline int clamp_index(int i, int maxv)
{
    if (i < 0)
        return 0;
    if (i >= maxv)
        return maxv - 1;
    return i;
}

// ---------- (1) Bicubic Interpolation (Keys' cubic, a = -0.5) ----------
// 가중치 커널: |x|<=1, 1<|x|<2 에 대해 다항식. Catmull-Rom(a=-0.5) 사용.
// 보간 포인트 x = x0 + fx (0<=fx<1) 주변의 4개 샘플을 쓰므로, 인덱스 셋은
// [x0-1, x0, x0+1, x0+2]가 된다. 경계는 복제(clamp).
static inline double cubic_kernel(double x, double a = -0.5)
{
    x = abs(x);
    if (x <= 1.0)
    {
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    }
    else if (x < 2.0)
    {
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    }
    else
    {
        return 0.0;
    }
}

// src 좌표계의 실수 위치 (ys, xs)에서 바이큐빅 샘플(그레이스케일)
static inline double sample_bicubic_gray(const Mat& src, double ys, double xs)
{
    const int h = src.rows, w = src.cols;
    const int y0 = static_cast<int>(floor(ys));
    const int x0 = static_cast<int>(floor(xs));
    const double fy = ys - y0; // [0,1)
    const double fx = xs - x0;

    double wy[4], wx[4];
    // 이웃 인덱스 m,n ∈ {-1,0,1,2}에 대해 거리 |f - n|, |f - m|을 커널에 넣는다.
    for (int m = -1; m <= 2; ++m)
        wy[m + 1] = cubic_kernel(fy - m);
    for (int n = -1; n <= 2; ++n)
        wx[n + 1] = cubic_kernel(fx - n);

    double sum = 0.0;
    for (int m = -1; m <= 2; ++m)
    {
        const int yy = clamp_index(y0 + m, h);
        const uint8_t* rowp = src.ptr<uint8_t>(yy);
        for (int n = -1; n <= 2; ++n)
        {
            const int xx = clamp_index(x0 + n, w);
            const double wgt = wy[m + 1] * wx[n + 1];
            sum += wgt * rowp[xx];
        }
    }
    // 값은 보간으로 인해 약간 벗어날 수 있으므로 0..255로 클램프
    return clamp(sum, 0.0, 255.0);
}

// 크기 변경(그레이스케일 전용). 매핑: (x_d+0.5)/sx - 0.5, OpenCV 방식과 동일.
Mat resize_bicubic_gray(const Mat& src, int new_w, int new_h)
{
    CV_Assert(src.type() == CV_8UC1);
    Mat dst(new_h, new_w, CV_8UC1);

    const double sx = static_cast<double>(src.cols) / new_w;
    const double sy = static_cast<double>(src.rows) / new_h;

    for (int y = 0; y < new_h; ++y)
    {
        double ys = (y + 0.5) * sy - 0.5;
        for (int x = 0; x < new_w; ++x)
        {
            double xs = (x + 0.5) * sx - 0.5;
            dst.at<uint8_t>(y, x) = static_cast<uint8_t>(round(sample_bicubic_gray(src, ys, xs)));
        }
    }
    return dst;
}

// ---------- 히스토그램/그래프 공통 ----------

// 채널 수(1 또는 3)에 대해 히스토그램(각 256-bin) 계산
// color의 경우 "각 color별로 독립" 원칙에 따라 3개 배열을 만든다.
vector<array<int, 256>> compute_hist(const Mat& img)
{
    const int ch = img.channels();
    CV_Assert(ch == 1 || ch == 3);

    vector<array<int, 256>> H(ch);
    for (int c = 0; c < ch; ++c)
        H[c].fill(0);

    const int h = img.rows, w = img.cols;

    if (ch == 1)
    {
        for (int y = 0; y < h; ++y)
        {
            const uint8_t* p = img.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
                H[0][p[x]]++;
        }
    }
    else
    {
        for (int y = 0; y < h; ++y)
        {
            const Vec3b* p = img.ptr<Vec3b>(y);
            for (int x = 0; x < w; ++x)
            {
                H[0][p[x][0]]++; // B
                H[1][p[x][1]]++; // G
                H[2][p[x][2]]++; // R
            }
        }
    }
    return H;
}

// CDF(누적분포) 계산. 정규화해서 [0,1] 범위.
static inline array<double, 256> hist_to_cdf(const array<int, 256>& H, int total_pixels)
{
    array<double, 256> cdf{};
    long long cum = 0;
    for (int i = 0; i < 256; ++i)
    {
        cum += H[i];
        cdf[i] = static_cast<double>(cum) / static_cast<double>(total_pixels);
    }
    return cdf;
}

// 막대 히스토그램 시각화(흑백). 입력은 단일 채널용 256-bin.
Mat draw_histogram(const array<int, 256>& H, int width = 512, int height = 300)
{
    Mat canvas(height, width, CV_8UC1, Scalar(255));
    const int bins = 256;
    int maxv = 0;
    for (int i = 0; i < bins; ++i)
        maxv = max(maxv, H[i]);
    if (maxv == 0)
        maxv = 1;

    const double sx = static_cast<double>(width) / bins;
    for (int i = 0; i < bins; ++i)
    {
        int h = static_cast<int>(round((static_cast<double>(H[i]) / maxv) * (height - 10)));
        int x0 = static_cast<int>(round(i * sx));
        int x1 = static_cast<int>(round((i + 1) * sx)) - 1;
        x1 = max(x1, x0);
        rectangle(canvas, Point(x0, height - 1 - h), Point(x1, height - 1), Scalar(0), FILLED);
    }
    return canvas;
}

// ---------- (2) Histogram Equalization ----------
// 채널 독립적으로 수행(B, G, R 각각에 대해 LUT 생성/적용)
Mat histogram_equalize(const Mat& src)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    const int ch = src.channels();
    const int N = src.rows * src.cols;

    auto Hs = compute_hist(src);

    // 채널별 LUT 계산
    vector<array<uint8_t, 256>> LUT(ch);
    for (int c = 0; c < ch; ++c)
    {
        auto cdf = hist_to_cdf(Hs[c], N);
        // 전형적인 equalize: y = round(255 * CDF(x))
        for (int i = 0; i < 256; ++i)
        {
            LUT[c][i] = static_cast<uint8_t>(round(255.0 * cdf[i]));
        }
    }

    // LUT 적용
    Mat dst(src.size(), src.type());
    if (ch == 1)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const uint8_t* sp = src.ptr<uint8_t>(y);
            uint8_t* dp = dst.ptr<uint8_t>(y);
            for (int x = 0; x < src.cols; ++x)
                dp[x] = LUT[0][sp[x]];
        }
    }
    else
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec3b* sp = src.ptr<Vec3b>(y);
            Vec3b* dp = dst.ptr<Vec3b>(y);
            for (int x = 0; x < src.cols; ++x)
            {
                dp[x][0] = LUT[0][sp[x][0]];
                dp[x][1] = LUT[1][sp[x][1]];
                dp[x][2] = LUT[2][sp[x][2]];
            }
        }
    }
    return dst;
}

// ---------- (3) Histogram Matching (Piecewise-linear target PDF) ----------
// 사용자 지정 분포: 세 점 (0, y0) - (mid, ymid) - (255, y255)를 잇는 2개의 선분.
// 보통 슬라이드의 "삼각형" 분포를 의미하므로 기본 y0=0, ymid=1, y255=0 로 둔다.
// 절차: 1) 원본 CDF 계산  2) 목표 PDF→CDF 계산  3) 매칭 LUT[g] = argmin_t |CDF_tgt[t]-CDF_src[g]|
array<double, 256> build_piecewise_pdf(int mid, double y0, double ymid, double y255)
{
    mid = clamp(mid, 0, 255);
    array<double, 256> pdf{};
    // 왼쪽(0..mid)
    for (int x = 0; x <= mid; ++x)
    {
        double t = (mid == 0) ? 1.0 : (static_cast<double>(x) / mid);
        pdf[x] = (1.0 - t) * y0 + t * ymid;
    }
    // 오른쪽(mid..255)
    for (int x = mid; x < 256; ++x)
    {
        double t = (255 == mid) ? 1.0 : (static_cast<double>(x - mid) / (255 - mid));
        pdf[x] = (1.0 - t) * ymid + t * y255;
    }
    // 음수 방지 및 정규화
    for (int i = 0; i < 256; ++i)
        pdf[i] = max(0.0, pdf[i]);
    double s = 0.0;
    for (int i = 0; i < 256; ++i)
        s += pdf[i];
    if (s <= 1e-12)
    { // 전부 0이면 균등분포로 대체
        for (int i = 0; i < 256; ++i)
            pdf[i] = 1.0;
        s = 256.0;
    }
    for (int i = 0; i < 256; ++i)
        pdf[i] /= s;
    return pdf;
}

array<double, 256> pdf_to_cdf(const array<double, 256>& pdf)
{
    array<double, 256> cdf{};
    double cum = 0.0;
    for (int i = 0; i < 256; ++i)
    {
        cum += pdf[i];
        cdf[i] = cum;
    }
    // 수치 오차로 1.0을 약간 넘길 수 있음 → 클램프
    for (int i = 0; i < 256; ++i)
        cdf[i] = min(1.0, max(0.0, cdf[i]));
    return cdf;
}

// src(1ch 또는 3ch)의 각 채널을 동일한 목표 CDF로 매칭
Mat histogram_match_piecewise(const Mat& src, int mid = 128,
    double y0 = 0.0, double ymid = 1.0, double y255 = 0.0)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    const int ch = src.channels();
    const int N = src.rows * src.cols;

    auto Hs = compute_hist(src);
    array<double, 256> tgt_pdf = build_piecewise_pdf(mid, y0, ymid, y255);
    array<double, 256> tgt_cdf = pdf_to_cdf(tgt_pdf);

    // 채널별 LUT 구축
    vector<array<uint8_t, 256>> LUT(ch);
    for (int c = 0; c < ch; ++c)
    {
        auto src_cdf = hist_to_cdf(Hs[c], N);
        // 매칭: src_cdf[g]와 가장 가까운 tgt_cdf[t]의 t를 g에 대응
        for (int g = 0; g < 256; ++g)
        {
            double sprob = src_cdf[g];
            // 이진 탐색
            int lo = 0, hi = 255;
            while (lo < hi)
            {
                int midx = (lo + hi) >> 1;
                if (tgt_cdf[midx] < sprob)
                    lo = midx + 1;
                else
                    hi = midx;
            }
            LUT[c][g] = static_cast<uint8_t>(lo);
        }
    }

    // LUT 적용
    Mat dst(src.size(), src.type());
    if (ch == 1)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const uint8_t* sp = src.ptr<uint8_t>(y);
            uint8_t* dp = dst.ptr<uint8_t>(y);
            for (int x = 0; x < src.cols; ++x)
                dp[x] = LUT[0][sp[x]];
        }
    }
    else
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec3b* sp = src.ptr<Vec3b>(y);
            Vec3b* dp = dst.ptr<Vec3b>(y);
            for (int x = 0; x < src.cols; ++x)
            {
                dp[x][0] = LUT[0][sp[x][0]];
                dp[x][1] = LUT[1][sp[x][1]];
                dp[x][2] = LUT[2][sp[x][2]];
            }
        }
    }
    return dst;
}

// ---------- 메인: I/O 전담 ----------
// argv:
//   [1] 입력 이미지 경로 (문제 1은 grayscale로 읽음)
//   [2] (옵션) bicubic scale, 기본 2.0
//   [3] (옵션) matching apex(mid), 기본 128
//   [4] (옵션) matching y0 (좌단 높이), 기본 0.0
//   [5] (옵션) matching y255 (우단 높이), 기본 0.0
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0]
            << " <input_image_path> [scale=2.0] [y0=0.0] [match_mid=128] [y255=0.0]\n";
        return 1;
    }
    const string in_path = argv[1];
    const double scale = (argc >= 3) ? stod(argv[2]) : 2.0;
    const double match_y0 = (argc >= 4) ? stod(argv[3]) : 0.0;
    const int match_mid = (argc >= 4) ? stoi(argv[4]) : 128;
    const double match_y255 = (argc >= 6) ? stod(argv[5]) : 0.0;

    // (1) Bicubic (입력: 흑백으로 읽기)
    Mat gray = imread(in_path, IMREAD_GRAYSCALE);
    imshow("origin_gray", gray);
    if (gray.empty())
    {
        cerr << "Failed to read image: " << in_path << "\n";
        return 1;
    }

    int new_w = max(1, static_cast<int>(round(gray.cols * scale)));
    int new_h = max(1, static_cast<int>(round(gray.rows * scale)));
    Mat bicubic = resize_bicubic_gray(gray, new_w, new_h);


    // (2) Histogram Equalization (입력: 컬러/그레이 그대로 읽기)
    Mat src_color = imread(in_path, IMREAD_UNCHANGED);
    imshow("origin_color", src_color);
    if (src_color.empty())
    {
        cerr << "Failed to read image (again): " << in_path << "\n";
        return 1;
    }
    if (src_color.type() != CV_8UC1 && src_color.type() != CV_8UC3)
    {
        cerr << "Only 8-bit 1ch/3ch images are supported.\n";
        return 1;
    }
    Mat eq = histogram_equalize(src_color);


    // 히스토그램 그래프(단일 채널 기준으로 저장: 그레이면 그대로, 컬러면 B 채널 그래프를 샘플로 저장)
    auto H_eq = compute_hist(eq);


    // (3) Histogram Matching (피크 위치와 끝단 높이는 인자로)
    Mat matched = histogram_match_piecewise(src_color, match_mid, match_y0, 1.0, match_y255);

    auto H_mt = compute_hist(matched);


    // 필요하면 주석 해제하여 시각화
    imshow("Bicubic", bicubic);
    imshow("Equalized", eq);
    imshow("Matched", matched);
    imshow("Equalized Hist (ch0)", draw_histogram(H_eq[0]));
    imshow("Matched Hist (ch0)", draw_histogram(H_mt[0]));

    waitKey(0);

    cout << "Done.\n";

    return 0;
}
