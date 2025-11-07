// 2.cpp
// 빌드 예: g++ -std=c++17 2.cpp -o run2 `pkg-config --cflags --libs opencv4`
// 요구사항: 침식/팽창/오프닝/클로징 직접 구현(이웃 내 min/max). 사각형 구조요소. 크기 증가별 결과 저장.
// 참고: 슬라이드의 수학적 정의와 "이웃 중 최소/최대값 대체" 설명. :contentReference[oaicite:5]{index=5}

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;

static const string INPUT_PATH_1 = "b1.png";
static const string INPUT_PATH_2 = "b2.png";
static const string OUT_DIR = "실행결과/";

// 가장자리 처리는 경계복제(replicate). 좌표 클램프.
inline int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

// BGR -> Gray (직접 변환: ITU-R BT.601 근사)
static Mat bgr2gray_manual(const Mat& src)
{
    Mat g(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
    {
        const Vec3b* sp = src.ptr<Vec3b>(y);
        uint8_t* gp = g.ptr<uint8_t>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            float B = sp[x][0], G = sp[x][1], R = sp[x][2];
            int Y = (int) round(0.114f * B + 0.587f * G + 0.299f * R);
            gp[x] = (uint8_t) min(255, max(0, Y));
        }
    }
    return g;
}

// Gray Erode: 윈도 내 최소값
static Mat erodeGray(const Mat& src, int kH, int kW)
{
    CV_Assert(src.type() == CV_8UC1);
    Mat dst(src.size(), CV_8UC1);
    int ah = kH / 2, aw = kW / 2;
    for (int y = 0; y < src.rows; ++y)
    {
        uint8_t* dp = dst.ptr<uint8_t>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            int mn = 255;
            for (int dy = -ah; dy <= ah; ++dy)
            {
                int yy = clampi(y + dy, 0, src.rows - 1);
                const uint8_t* sp = src.ptr<uint8_t>(yy);
                for (int dx = -aw; dx <= aw; ++dx)
                {
                    int xx = clampi(x + dx, 0, src.cols - 1);
                    mn = min(mn, (int) sp[xx]);
                }
            }
            dp[x] = (uint8_t) mn;
        }
    }
    return dst;
}

// Gray Dilate: 윈도 내 최대값
static Mat dilateGray(const Mat& src, int kH, int kW)
{
    CV_Assert(src.type() == CV_8UC1);
    Mat dst(src.size(), CV_8UC1);
    int ah = kH / 2, aw = kW / 2;
    for (int y = 0; y < src.rows; ++y)
    {
        uint8_t* dp = dst.ptr<uint8_t>(y);
        for (int x = 0; x < src.cols; ++x)
        {
            int mx = 0;
            for (int dy = -ah; dy <= ah; ++dy)
            {
                int yy = clampi(y + dy, 0, src.rows - 1);
                const uint8_t* sp = src.ptr<uint8_t>(yy);
                for (int dx = -aw; dx <= aw; ++dx)
                {
                    int xx = clampi(x + dx, 0, src.cols - 1);
                    mx = max(mx, (int) sp[xx]);
                }
            }
            dp[x] = (uint8_t) mx;
        }
    }
    return dst;
}

static Mat opening(const Mat& src, int kH, int kW) { return dilateGray(erodeGray(src, kH, kW), kH, kW); } // A∘B=(A⊖B)⊕B :contentReference[oaicite:6]{index=6}
static Mat closing(const Mat& src, int kH, int kW) { return erodeGray(dilateGray(src, kH, kW), kH, kW); } // A∙B=(A⊕B)⊖B :contentReference[oaicite:7]{index=7}

int main()
{
    filesystem::create_directories(OUT_DIR);

    Mat A_srcColor = imread(INPUT_PATH_1, IMREAD_COLOR);
    imshow("A_original", A_srcColor);

    Mat A_src = bgr2gray_manual(A_srcColor);

    vector<pair<int, int>> A_sizes = { {3,3}, {5,5}, {9,9} };
    for (auto [kh, kw] : A_sizes)
    {
        Mat A_op = opening(A_src, kh, kw);
        Mat A_cl = closing(A_src, kh, kw);

        imshow("A_opening_" + to_string(kh) + "x" + to_string(kw), A_op);
        imshow("A_closing_" + to_string(kh) + "x" + to_string(kw), A_cl);

        imwrite("A_" + OUT_DIR + "opening_" + to_string(kh) + "x" + to_string(kw) + ".png", A_op);
        imwrite("A_" + OUT_DIR + "closing_" + to_string(kh) + "x" + to_string(kw) + ".png", A_cl);
    }

    Mat B_srcColor = imread(INPUT_PATH_2, IMREAD_COLOR);
    imshow("B_original", B_srcColor);

    Mat B_src = bgr2gray_manual(B_srcColor);

    vector<pair<int, int>> B_sizes = { {3,3}, {5,5}, {9,9} };
    for (auto [kh, kw] : B_sizes)
    {
        Mat B_op = opening(B_src, kh, kw);
        Mat B_cl = closing(B_src, kh, kw);

        imshow("B_opening_" + to_string(kh) + "x" + to_string(kw), B_op);
        imshow("B_closing_" + to_string(kh) + "x" + to_string(kw), B_cl);

        imwrite("B_" + OUT_DIR + "opening_" + to_string(kh) + "x" + to_string(kw) + ".png", B_op);
        imwrite("B_" + OUT_DIR + "closing_" + to_string(kh) + "x" + to_string(kw) + ".png", B_cl);
    }

    cout << "[완료] out2/ 에 결과 저장\n";

    waitKey(0);

    return 0;
}
