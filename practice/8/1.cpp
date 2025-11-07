// 1.cpp
// 빌드 예: g++ -std=c++17 1.cpp -o run1 `pkg-config --cflags --libs opencv4`
// 요구사항: 이미지 입출력만 OpenCV 사용. 나머지(색공간변환, 평활화, 픽셀처리)는 직접 구현.
// 상수/경로는 하드코딩.

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

// ------------------------------- 하드코딩 상수/경로 -------------------------------
static const string INPUT_PATH = "lena.png";   // BGR, 512x512 가정(슬라이드 명시)
static const string OUT_DIR = "실행결과/";

// Piecewise-Linear T(r) 파라미터 (0~255 구간 3분할)
// 슬라이드의 정의: (r1,s1), (r2,s2)로 구간별 선형 매핑. :contentReference[oaicite:2]{index=2}
static const int r1 = 70, s1 = 30;
static const int r2 = 180, s2 = 220;

// ------------------------------- 유틸 -------------------------------
inline uint8_t clamp_u8(int v) { return (uint8_t) min(255, max(0, v)); }
inline float clamp01(float x) { return min(1.0f, max(0.0f, x)); }
inline float deg2rad(float d) { return d * (float) CV_PI / 180.0f; }
inline float rad2deg(float r) { return r * 180.0f / (float) CV_PI; }

// 구간별 선형 스트레칭 LUT 생성 (0..255)
static vector<uint8_t> build_piecewise_LUT()
{
    vector<uint8_t> lut(256);
    for (int r = 0; r < 256; ++r)
    {
        float s = 0.f;
        if (r <= r1)
        {
            s = (r1 == 0) ? s1 : ((float) s1 / (float) r1) * (float) r;
        }
        else if (r < r2)
        {
            float t = (float) (r - r1) / (float) (r2 - r1);
            s = (float) s1 + ((float) (s2 - s1)) * t;
        }
        else
        {
            float t = (float) (r - r2) / (float) (255 - r2);
            s = (float) s2 + ((float) (255 - s2)) * t;
        }
        lut[r] = clamp_u8((int) round(s));
    }
    return lut;
}

// BGR(0..255) -> HSI (H:0..360, S:0..1, I:0..1)  [슬라이드 11–16의 수식 사용] :contentReference[oaicite:3]{index=3}
static void bgr2hsi(const Vec3b& bgr, float& H_deg, float& S, float& I)
{
    float B = bgr[0] / 255.0f, G = bgr[1] / 255.0f, R = bgr[2] / 255.0f;
    float num = 0.5f * ((R - G) + (R - B));
    float den = sqrt((R - G) * (R - G) + (R - B) * (G - B)) + 1e-12f; // 0 나눔 방지
    float theta = acos(max(-1.0f, min(1.0f, num / den))); // 라디안

    if (B <= G) H_deg = rad2deg(theta);
    else       H_deg = 360.0f - rad2deg(theta);

    float minRGB = min(R, min(G, B));
    I = (R + G + B) / 3.0f;
    if (I <= 1e-12f) { S = 0.0f; H_deg = 0.0f; return; } // 완전 흑색
    S = 1.0f - (minRGB / I);
    S = clamp01(S);
    if (isnan(H_deg)) H_deg = 0.0f;
}

// HSI -> BGR (슬라이드 16의 3구간 공식 형태를 구현) :contentReference[oaicite:4]{index=4}
static Vec3b hsi2bgr(float H_deg, float S, float I)
{
    H_deg = fmodf(H_deg, 360.0f); if (H_deg < 0) H_deg += 360.0f;
    S = clamp01(S); I = clamp01(I);

    float R = 0, G = 0, B = 0;
    auto cdeg = [](float d) { return cos(deg2rad(d)); };

    if (0.0f <= H_deg && H_deg < 120.0f)
    {
        float Bc = I * (1.0f - S);
        float Rn = I * (1.0f + (S * cdeg(H_deg)) / (cdeg(60.0f - H_deg) + 1e-12f));
        float Gn = 3.0f * I - (Rn + Bc);
        R = Rn; G = Gn; B = Bc;
    }
    else if (120.0f <= H_deg && H_deg < 240.0f)
    {
        float Ht = H_deg - 120.0f;
        float Rc = I * (1.0f - S);
        float Gn = I * (1.0f + (S * cdeg(Ht)) / (cdeg(60.0f - Ht) + 1e-12f));
        float Bn = 3.0f * I - (Rc + Gn);
        R = Rc; G = Gn; B = Bn;
    }
    else
    { // 240° ≤ H < 360°
        float Ht = H_deg - 240.0f;
        float Gc = I * (1.0f - S);
        float Bn = I * (1.0f + (S * cdeg(Ht)) / (cdeg(60.0f - Ht) + 1e-12f));
        float Rn = 3.0f * I - (Gc + Bn);
        R = Rn; G = Gc; B = Bn;
    }
    int r = (int) round(R * 255.0f);
    int g = (int) round(G * 255.0f);
    int b = (int) round(B * 255.0f);
    return Vec3b(clamp_u8(b), clamp_u8(g), clamp_u8(r));
}

// I 채널(0..1)을 0..255 정수화 후 히스토그램 평활화
static void equalize_I_channel_inplace(vector<vector<float>>& H, vector<vector<float>>& S, vector<vector<float>>& I)
{
    const int Ht = (int) I.size();
    const int W = (int) I[0].size();
    vector<int> hist(256, 0);

    // 히스토그램
    for (int y = 0; y < Ht; ++y) for (int x = 0; x < W; ++x)
    {
        int level = (int) round(clamp01(I[y][x]) * 255.0f);
        hist[level]++;
    }
    // CDF
    vector<int> cdf(256, 0);
    int run = 0, total = Ht * W;
    for (int i = 0; i < 256; ++i) { run += hist[i]; cdf[i] = run; }

    // 매핑: I' = CDF / total
    for (int y = 0; y < Ht; ++y) for (int x = 0; x < W; ++x)
    {
        int level = (int) round(clamp01(I[y][x]) * 255.0f);
        float newI = (float) cdf[level] / (float) total;
        I[y][x] = clamp01(newI);
    }
}

int main()
{
    filesystem::create_directories(OUT_DIR);

    Mat src = imread(INPUT_PATH, IMREAD_COLOR);
    imshow("original", src);

    if (src.empty()) { cerr << "입력 이미지를 열 수 없습니다: " << INPUT_PATH << "\n"; return 1; }

    const int Ht = src.rows, W = src.cols;

    // -------------------- Programming #1: Piecewise-Linear Contrast Stretching --------------------
    auto lut = build_piecewise_LUT();
    Mat stretched(src.size(), CV_8UC3);
    for (int y = 0; y < Ht; ++y)
    {
        const uint8_t* sp = src.ptr<uint8_t>(y);
        uint8_t* dp = stretched.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x)
        {
            uint8_t b = sp[3 * x + 0], g = sp[3 * x + 1], r = sp[3 * x + 2];
            dp[3 * x + 0] = lut[b];
            dp[3 * x + 1] = lut[g];
            dp[3 * x + 2] = lut[r];
        }
    }
    imshow("1_piecewise_bgr", stretched);
    imwrite(OUT_DIR + "1_piecewise_bgr.png", stretched);

    // -------------------- Programming #2: RGB<->HSI + I-Equalization --------------------
    // BGR -> HSI
    vector<vector<float>> Hmap(Ht, vector<float>(W));
    vector<vector<float>> Smap(Ht, vector<float>(W));
    vector<vector<float>> Imap(Ht, vector<float>(W));
    for (int y = 0; y < Ht; ++y)
    {
        const Vec3b* sp = src.ptr<Vec3b>(y);
        for (int x = 0; x < W; ++x)
        {
            float H_deg, S, I;
            bgr2hsi(sp[x], H_deg, S, I);
            Hmap[y][x] = H_deg; Smap[y][x] = S; Imap[y][x] = I;
        }
    }
    // I 채널만 평활화
    equalize_I_channel_inplace(Hmap, Smap, Imap);
    // HSI -> BGR
    Mat hsi_eq(src.size(), CV_8UC3);
    for (int y = 0; y < Ht; ++y)
    {
        Vec3b* dp = hsi_eq.ptr<Vec3b>(y);
        for (int x = 0; x < W; ++x)
        {
            dp[x] = hsi2bgr(Hmap[y][x], Smap[y][x], Imap[y][x]);
        }
    }
    imshow("2_hsi_I_equalized", hsi_eq);
    imwrite(OUT_DIR + "2_hsi_I_equalized.png", hsi_eq);

    cout << "[완료] out1/ 에 결과 저장\n";

    waitKey(0);

    return 0;
}
