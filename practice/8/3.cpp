// 3.cpp
// Hough Transform (manual) + simple probabilistic-style segment extraction
// No OpenCV ops except I/O and basic container types.
// Build: g++ -std=c++17 3.cpp -o hough `pkg-config --cflags --libs opencv4`
// Usage:
//   ./hough input.jpg out_prefix --edge_thresh=100 --theta_bins=180 --rho_bins=0 --acc_thresh=120 --min_len=30 --max_gap=5
// Notes:
//   - If rho_bins=0, it is set to 2*ceil(hypot(W,H))
//   - Edge detector uses Sobel (|Gx|+|Gy|) then threshold
//   - Lines are detected by standard Hough voting; for each strong bin, we sample along the line
//     and group edge points with gap<=max_gap to form segments; segments with length>=min_len are drawn.
//   - Drawing uses our own Bresenham (no cv::line).
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <tuple>
#include <algorithm>

using namespace cv;
using namespace std;

static inline uint8_t gray_from_bgr(const Vec3b& bgr)
{
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

// Sobel |Gx| + |Gy| (3x3)
static Mat sobel_abs_sum(const Mat& g)
{
    static const int kx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
    static const int ky[3][3] = { {-1,-2,-1}, {0,0,0}, {1,2,1} };
    Mat out(g.size(), CV_16S);
    for (int y = 0; y < g.rows; ++y)
    {
        short* op = out.ptr<short>(y);
        for (int x = 0; x < g.cols; ++x)        
{
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = max(0, min(g.rows - 1, y + dy));
                const uint8_t* gp = g.ptr<uint8_t>(yy);
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = max(0, min(g.cols - 1, x + dx));
                    int v = gp[xx];
                    gx += v * kx[dy + 1][dx + 1];
                    gy += v * ky[dy + 1][dx + 1];
                }
            }
            op[x] = (short) (abs(gx) + abs(gy));
        }
    }
    return out;
}

static Mat threshold_u8_from_16s(const Mat& m, int th)
{
    Mat out(m.size(), CV_8UC1);
    for (int y = 0; y < m.rows; ++y)
    {
        const short* ip = m.ptr<short>(y);
        uint8_t* op = out.ptr<uint8_t>(y);
        for (int x = 0; x < m.cols; ++x)
        {
            op[x] = (ip[x] >= th) ? 255 : 0;
        }
    }
    return out;
}

// Simple Bresenham line draw on BGR image
static void draw_line_bgr(Mat& img, int x0, int y0, int x1, int y1, const Vec3b& color, int thickness = 1)
{
// Basic Bresenham for thickness=1; if thickness>1, draw multiple offset lines.
    auto plot = [&](int x, int y) {
        if (0 <= x && x < img.cols && 0 <= y && y < img.rows) img.ptr<Vec3b>(y)[x] = color;
        };
    auto bres = [&](int X0, int Y0, int X1, int Y1) {
        int dx = abs(X1 - X0), sx = X0 < X1 ? 1 : -1;
        int dy = -abs(Y1 - Y0), sy = Y0 < Y1 ? 1 : -1;
        int err = dx + dy, e2;
        while (true)
        {
            plot(X0, Y0);
            if (X0 == X1 && Y0 == Y1) break;
            e2 = 2 * err;
            if (e2 >= dy) { err += dy; X0 += sx; }
            if (e2 <= dx) { err += dx; Y0 += sy; }
        }
        };
    if (thickness <= 1) { bres(x0, y0, x1, y1); return; }
    // crude thickness by drawing shifted lines perpendicular to the segment
    double vx = x1 - x0, vy = y1 - y0;
    double len = sqrt(vx * vx + vy * vy) + 1e-9;
    double nx = -vy / len, ny = vx / len; // unit normal
    int half = thickness / 2;
    for (int t = -half; t <= half; ++t)    
{
        int sx0 = (int) round(x0 + nx * t);
        int sy0 = (int) round(y0 + ny * t);
        int sx1 = (int) round(x1 + nx * t);
        int sy1 = (int) round(y1 + ny * t);
        bres(sx0, sy0, sx1, sy1);
    }
}

// Parse "--key=value" (int) with default
static int get_int_arg(int argc, char** argv, const string& key, int def)
{
    string pre = "--" + key + "=";
    for (int i = 1; i < argc; ++i)
    {
        string s = argv[i];
        if (s.rfind(pre, 0) == 0) return stoi(s.substr(pre.size()));
    }
    return def;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " input.jpg out_prefix [--edge_thresh=100] [--theta_bins=180] [--rho_bins=0] [--acc_thresh=120] [--min_len=30] [--max_gap=5]\n";
        return 1;
    }
    string in_path = argv[1];
    string out_prefix = argv[2];
    int edge_th = get_int_arg(argc, argv, "edge_thresh", 100);
    int theta_bins = get_int_arg(argc, argv, "theta_bins", 180);
    int rho_bins = get_int_arg(argc, argv, "rho_bins", 0);
    int acc_th = get_int_arg(argc, argv, "acc_thresh", 120);
    int min_len = get_int_arg(argc, argv, "min_len", 30);
    int max_gap = get_int_arg(argc, argv, "max_gap", 5);

    Mat src = imread(in_path, IMREAD_COLOR);
    if (src.empty()) { cerr << "Failed to read: " << in_path << "\n"; return 2; }
    Mat g = to_gray(src);
    Mat mag = sobel_abs_sum(g);
    Mat edges = threshold_u8_from_16s(mag, edge_th);
    imwrite(out_prefix + "_edges.png", edges);

    const int W = src.cols, H = src.rows;
    double rho_max = hypot((double) W, (double) H);
    if (rho_bins <= 0) rho_bins = (int) (2.0 * ceil(rho_max));
    vector<int> acc((size_t) theta_bins * (size_t) rho_bins, 0);

    auto acc_at = [&](int ti, int ri)->int& {
        return acc[(size_t) ti * (size_t) rho_bins + (size_t) ri];
        };

        // Precompute theta values
    vector<double> thetas(theta_bins);
    for (int t = 0; t < theta_bins; ++t)
    {
        thetas[t] = (M_PI * t) / (double) theta_bins; // [0, pi)
    }
    double drho = (2.0 * rho_max) / (double) rho_bins;

    // Vote
    for (int y = 0; y < H; ++y)
    {
        const uint8_t* ep = edges.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x)
        {
            if (ep[x] == 0) continue;
            for (int t = 0; t < theta_bins; ++t)
            {
                double th = thetas[t];
                double rho = x * cos(th) + y * sin(th);
                int ri = (int) floor((rho + rho_max) / drho);
                if (ri >= 0 && ri < rho_bins) acc_at(t, ri)++;
            }
        }
    }

    // Collect peaks
    struct Peak { int ti; int ri; int votes; };
    vector<Peak> peaks;
    for (int t = 0; t < theta_bins; ++t)    
{
        for (int r = 0; r < rho_bins; ++r)
        {
            int v = acc_at(t, r);
            if (v >= acc_th) peaks.push_back({ t,r,v });
        }
    }
    // Optional: sort by votes desc
    sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return a.votes > b.votes; });

    // Draw segments on a copy
    Mat vis = src.clone();
    int drawn_segments = 0;
    for (const auto& pk : peaks)
    {
        double th = thetas[pk.ti];
        double rho = -rho_max + (pk.ri + 0.5) * drho; // center of bin
        // Closest point to origin on the line
        double cx = rho * cos(th);
        double cy = rho * sin(th);
        // Direction along the line
        double dx = -sin(th);
        double dy = cos(th);
        // Sample along t
        double tmin = -max(W, H) * 1.5;
        double tmax = max(W, H) * 1.5;
        double tstep = 1.0;

        // Collect edge hits along t
        vector<int> hits;
        vector<Point> hit_pts;
        for (double t = tmin; t <= tmax; t += tstep)
        {
            int xx = (int) round(cx + dx * t);
            int yy = (int) round(cy + dy * t);
            if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
            if (edges.ptr<uint8_t>(yy)[xx])
            {
                hits.push_back((int) round(t));
                hit_pts.emplace_back(xx, yy);
            }
        }
        if (hits.empty()) continue;
        // Group by gap
        int start_idx = 0;
        for (size_t i = 1; i <= hits.size(); ++i)
        {
            bool split = (i == hits.size()) || (hits[i] - hits[i - 1] > max_gap);
            if (split)            
{
                int seg_len = hits[i - 1] - hits[start_idx];
                if (seg_len >= min_len)
                {
// endpoints
                    Point p0 = hit_pts[start_idx];
                    Point p1 = hit_pts[i - 1];
                    draw_line_bgr(vis, p0.x, p0.y, p1.x, p1.y, Vec3b(0, 0, 255), 2);
                    drawn_segments++;
                }
                start_idx = (int) i;
            }
        }
    }

    string out_path = out_prefix + "_hough.png";
    imwrite(out_path, vis);
    cout << "Edges: " << out_prefix << "_edges.png\n";
    cout << "Hough visualization saved: " << out_path << "\n";
    cout << "Detected peaks: " << peaks.size() << ", drawn segments: " << drawn_segments << "\n";
    cout << "Parameters: edge_thresh=" << edge_th << ", theta_bins=" << theta_bins << ", rho_bins=" << rho_bins
        << ", acc_thresh=" << acc_th << ", min_len=" << min_len << ", max_gap=" << max_gap << "\n";
    return 0;
}
