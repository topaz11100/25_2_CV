// 1.cpp
// Piecewise-Linear Contrast Stretching (no OpenCV ops except I/O)
// Build: g++ -std=c++17 1.cpp -o pwl `pkg-config --cflags --libs opencv4`
// Usage:
//   ./pwl input.jpg output.jpg r1 s1 r2 s2 [--mode=channel|luma]
// Notes:
//   - r1,s1,r2,s2 are 0..255 (integers), with 0 <= r1 < r2 <= 255 and 0 <= s1 < s2 <= 255
//   - mode=channel  : apply PWL per B,G,R channel independently (default)
//   - mode=luma     : apply PWL on luminance Y (0.114B+0.587G+0.299R) and rescale BGR by Y ratio
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
using namespace cv;
using namespace std;

static inline uint8_t clipu8(int v) { return (uint8_t) min(255, max(0, v)); }
static inline double clampd(double v, double lo, double hi) { return max(lo, min(hi, v)); }

static inline uint8_t pwl_map(uint8_t r, int r1, int s1, int r2, int s2)
{
// 3-piece linear mapping
    if (r <= r1)
    {
        if (r1 == 0) return (uint8_t) s1; // degenerate: map all to s1
        double m = (double) s1 / (double) r1;
        return clipu8((int) round(m * r));
    }
    else if (r <= r2)
    {
        double m = (double) (s2 - s1) / (double) (r2 - r1);
        return clipu8((int) round(s1 + m * (r - r1)));
    }
    else
    {
        if (r2 == 255) return (uint8_t) s2; // degenerate: map all to s2
        double m = (double) (255 - s2) / (double) (255 - r2);
        return clipu8((int) round(s2 + m * (r - r2)));
    }
}

static inline double luminance_from_bgr(const Vec3b& bgr)
{
// ITU-R BT.601 luma (approx): Y = 0.114B + 0.587G + 0.299R
    return 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2];
}

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        cerr << "Usage: " << argv[0] << " input.jpg output.jpg r1 s1 r2 s2 [--mode=channel|luma]\n";
        return 1;
    }
    string in_path = argv[1];
    string out_path = argv[2];
    int r1 = stoi(argv[3]);
    int s1 = stoi(argv[4]);
    int r2 = stoi(argv[5]);
    int s2 = stoi(argv[6]);
    string mode = "channel";
    if (argc >= 8)
    {
        string opt = argv[7];
        if (opt.rfind("--mode=", 0) == 0) mode = opt.substr(7);
    }
    if (!(0 <= r1 && r1 < r2 && r2 <= 255 && 0 <= s1 && s1 < s2 && s2 <= 255))
    {
        cerr << "Invalid breakpoints. Require 0 <= r1 < r2 <= 255 and 0 <= s1 < s2 <= 255\n";
        return 2;
    }

    Mat src = imread(in_path, IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "Failed to read: " << in_path << "\n";
        return 3;
    }
    Mat dst(src.size(), CV_8UC3);

    if (mode == "luma")    
{
// PWL on Y then scale BGR by ratio sY / Y (preserve chroma approximately)
        const double eps = 1e-6;
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec3b* sp = src.ptr<Vec3b>(y);
            Vec3b* dp = dst.ptr<Vec3b>(y);
            for (int x = 0; x < src.cols; ++x)
            {
                double Y = luminance_from_bgr(sp[x]);
                uint8_t sY = pwl_map((uint8_t) round(Y), r1, s1, r2, s2);
                double ratio = sY / max(eps, Y);
                // scale each channel by ratio and clip
                for (int c = 0; c < 3; ++c)                
{
                    dp[x][c] = clipu8((int) round(sp[x][c] * ratio));
                }
            }
        }
    }
    else
    {
        // per-channel PWL (default)
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec3b* sp = src.ptr<Vec3b>(y);
            Vec3b* dp = dst.ptr<Vec3b>(y);
            for (int x = 0; x < src.cols; ++x)
            {
                dp[x][0] = pwl_map(sp[x][0], r1, s1, r2, s2);
                dp[x][1] = pwl_map(sp[x][1], r1, s1, r2, s2);
                dp[x][2] = pwl_map(sp[x][2], r1, s1, r2, s2);
            }
        }
    }

    if (!imwrite(out_path, dst))
    {
        cerr << "Failed to write: " << out_path << "\n";
        return 4;
    }
    cout << "Saved: " << out_path << "\n";
    return 0;
}
