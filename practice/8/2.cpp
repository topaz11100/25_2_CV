// 2.cpp
// Morphological Processing (grayscale): erosion, dilation, opening, closing
// Implemented without OpenCV morphology (only I/O).
// Build: g++ -std=c++17 2.cpp -o morph `pkg-config --cflags --libs opencv4`
// Usage:
//   ./morph input.jpg out_prefix op ksize_list
// Examples:
//   ./morph lena.png out open 3,5,7
//   ./morph lena.png out close 5
// Notes:
//   - op in {erode, dilate, open, close}
//   - ksize_list: comma-separated odd integers >=3 (rectangular structuring element)
// Output files:
//   out_prefix_{op}_k{K}.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;

static inline uint8_t gray_from_bgr(const Vec3b& bgr)
{
// BT.601
    double Y = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2];
    int v = (int) round(Y);
    if (v < 0) v = 0; if (v > 255) v = 255;
    return (uint8_t) v;
}

static Mat to_gray(const Mat& bgr)
{
    Mat g(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
    {
        const Vec3b* sp = bgr.ptr<Vec3b>(y);
        uint8_t* gp = g.ptr<uint8_t>(y);
        for (int x = 0; x < bgr.cols; ++x) gp[x] = gray_from_bgr(sp[x]);
    }
    return g;
}

// replicate-border accessor
static inline uint8_t at_rep(const Mat& g, int y, int x)
{
    y = max(0, min(g.rows - 1, y));
    x = max(0, min(g.cols - 1, x));
    return g.ptr<uint8_t>(y)[x];
}

static Mat erode_gray(const Mat& g, int k)
{
    int r = k / 2;
    Mat out(g.size(), CV_8UC1);
    for (int y = 0; y < g.rows; ++y)
    {
        uint8_t* op = out.ptr<uint8_t>(y);
        for (int x = 0; x < g.cols; ++x)
        {
            int mn = 255;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    mn = min(mn, (int) at_rep(g, y + dy, x + dx));
            op[x] = (uint8_t) mn;
        }
    }
    return out;
}

static Mat dilate_gray(const Mat& g, int k)
{
    int r = k / 2;
    Mat out(g.size(), CV_8UC1);
    for (int y = 0; y < g.rows; ++y)
    {
        uint8_t* op = out.ptr<uint8_t>(y);
        for (int x = 0; x < g.cols; ++x)
        {
            int mx = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    mx = max(mx, (int) at_rep(g, y + dy, x + dx));
            op[x] = (uint8_t) mx;
        }
    }
    return out;
}

static Mat opening_gray(const Mat& g, int k)
{
    Mat e = erode_gray(g, k);
    return dilate_gray(e, k);
}

static Mat closing_gray(const Mat& g, int k)
{
    Mat d = dilate_gray(g, k);
    return erode_gray(d, k);
}

static vector<int> parse_ksizes(const string& s)
{
    vector<int> ks;
    string tmp; stringstream ss(s);
    while (getline(ss, tmp, ','))
    {
        if (tmp.empty()) continue;
        ks.push_back(stoi(tmp));
    }
    return ks;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " input.jpg out_prefix op ksize_list\n";
        return 1;
    }
    string in_path = argv[1];
    string out_prefix = argv[2];
    string op = argv[3];
    vector<int> ks = parse_ksizes(argv[4]);

    Mat src = imread(in_path, IMREAD_COLOR);
    if (src.empty()) { cerr << "Failed to read: " << in_path << "\n"; return 2; }
    Mat g = to_gray(src);

    for (int k : ks)
    {
        if (k < 3 || (k % 2) == 0)
        {
            cerr << "Skip invalid ksize=" << k << " (must be odd >=3)\n";
            continue;
        }
        Mat out;
        if (op == "erode") out = erode_gray(g, k);
        else if (op == "dilate") out = dilate_gray(g, k);
        else if (op == "open") out = opening_gray(g, k);
        else if (op == "close") out = closing_gray(g, k);
        else { cerr << "Unknown op: " << op << "\n"; return 3; }

        string out_path = out_prefix + "_" + op + "_k" + to_string(k) + ".png";
        if (!imwrite(out_path, out)) { cerr << "Failed to write: " << out_path << "\n"; return 4; }
        cout << "Saved: " << out_path << "\n";
    }
    return 0;
}
