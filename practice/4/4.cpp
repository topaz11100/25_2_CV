#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

// 입력 파일 경로
string src_path = "/home/yongokhan/바탕화면/25_2_CV/source/lena.png";
// 출력 폴더
string rst_path = "/home/yongokhan/바탕화면/25_2_CV/practice/4/실행결과/";

// 시드 고정 (재현성)
unsigned int seed = 42;
mt19937 gen(seed);

// ============================
// Padding 함수 (문제 #1 재사용)
// ============================
void pad_zero(Mat& img, int p)
{
    uchar* ptr = img.data;
    int step = img.step;
    int ch = img.channels();

    // 위쪽 p줄, 아래쪽 p줄을 0으로 채움
    for (int y = 0; y < p; ++y)
        memset(ptr + y * step, 0, step),
        memset(ptr + (img.rows - 1 - y) * step, 0, step);

    // 좌우 p픽셀 0으로 채움
    for (int y = p; y < img.rows - p; ++y)
    {
        uchar* row = ptr + y * step;
        for (int x = 0; x < p * ch; ++x) row[x] = 0;
        for (int x = (img.cols - p) * ch; x < img.cols * ch; ++x) row[x] = 0;
    }
}

// Replicate Padding (가장자리 복제)
void pad_replicate(Mat& img, int p)
{
    int ch = img.channels();
    int step = img.step;

    // 위/아래 줄 복제
    for (int y = 0; y < p; ++y)
    {
        uchar* top = img.ptr<uchar>(p);
        uchar* bot = img.ptr<uchar>(img.rows - p - 1);
        memcpy(img.ptr<uchar>(y), top, step);
        memcpy(img.ptr<uchar>(img.rows - 1 - y), bot, step);
    }

    // 좌우 복제
    for (int y = p; y < img.rows - p; ++y)
    {
        uchar* row = img.ptr<uchar>(y);
        for (int c = 0; c < ch; ++c)
        {
            uchar left_val = row[p * ch + c];
            uchar right_val = row[(img.cols - p - 1) * ch + c];
            for (int x = 0; x < p; ++x)
            {
                row[x * ch + c] = left_val;
                row[(img.cols - 1 - x) * ch + c] = right_val;
            }
        }
    }
}

// Reflect Padding (경계 제외 반사)
void pad_reflect(Mat& img, int p)
{
    int ch = img.channels();

    // 위/아래
    for (int y = 0; y < p; ++y)
    {
        uchar* top = img.ptr<uchar>(p + y + 1);    // 경계 제외
        uchar* bot = img.ptr<uchar>(img.rows - p - 2 - y);
        memcpy(img.ptr<uchar>(p - 1 - y), top, img.step);
        memcpy(img.ptr<uchar>(img.rows - p + y), bot, img.step);
    }
    // 좌우
    for (int y = 0; y < img.rows; ++y)
    {
        uchar* row = img.ptr<uchar>(y);
        for (int c = 0; c < ch; ++c)
        {
            for (int x = 0; x < p; ++x)
            {
                row[(p - 1 - x) * ch + c] = row[(p + 1 + x) * ch + c];
                row[(img.cols - p + x) * ch + c] = row[(img.cols - p - 2 - x) * ch + c];
            }
        }
    }
}

// Reflect_101 Padding (경계 포함 반사)
void pad_reflect_101(Mat& img, int p)
{
    int ch = img.channels();

    // 위/아래
    for (int y = 0; y < p; ++y)
    {
        uchar* top = img.ptr<uchar>(p + y);        // 경계 포함
        uchar* bot = img.ptr<uchar>(img.rows - p - 1 - y);
        memcpy(img.ptr<uchar>(p - 1 - y), top, img.step);
        memcpy(img.ptr<uchar>(img.rows - p + y), bot, img.step);
    }

    // 좌우
    for (int y = 0; y < img.rows; ++y)
    {
        uchar* row = img.ptr<uchar>(y);
        for (int c = 0; c < ch; ++c)
        {
            for (int x = 0; x < p; ++x)
            {
                row[(p - 1 - x) * ch + c] = row[(p + x) * ch + c];
                row[(img.cols - p + x) * ch + c] = row[(img.cols - p - 1 - x) * ch + c];
            }
        }
    }
}

// ============================
// 공통 유틸 (문제 #1 재사용)
// ============================
Mat padding(Mat& src, int p, void (*fill)(Mat&, int))
{
    Mat rst(src.size() + Size(2 * p, 2 * p), src.type());

    uchar* rst_P = rst.data;
    uchar* src_P = src.data;

    int l_rst_r = rst.step;
    int l_src_r = src.step;
    int ch = src.channels();

    // 원본 영역을 패딩된 영상의 중앙에 복사
    for (int y = 0; y < src.rows; ++y)
    {
        uchar* src_row = src_P + y * l_src_r;
        uchar* rst_row = rst_P + (y + p) * l_rst_r + p * ch;
        memcpy(rst_row, src_row, src.cols * ch * sizeof(uchar));
    }

    // 나머지 패딩 영역 채우기
    fill(rst, p);
    return rst;
}

/**
 * 이미 패딩이 된 이미지에 대해 정사각 커널 합성곱 (선형)
 * @tparam N : 커널 원소 자료형
 */
template<class N>
Mat apply_kernel(Mat& src, const Mat& ker)
{
    int p = ker.rows / 2;  // 패딩 반경
    int ch = src.channels();

    Mat rst(src.rows - 2 * p, src.cols - 2 * p, src.type());

    uchar* rst_P = rst.data;
    uchar* src_P = src.data;
    const N* ker_P = ker.ptr<N>();

    int l_rst_r = rst.step, l_src_r = src.step, l_ker_r = ker.step1();

    for (int y = 0; y < rst.rows; ++y)
        for (int x = 0; x < rst.cols; ++x)
            for (int c = 0; c < ch; ++c)
            {
                double v = 0.0;
                for (int ky = -p; ky <= p; ++ky)
                    for (int kx = -p; kx <= p; ++kx)
                        v += static_cast<double>(src_P[l_src_r * ((y + p) + ky) + ch * ((x + p) + kx) + c]) *
                        static_cast<double>(ker_P[l_ker_r * (p + ky) + (p + kx)]);
                rst_P[l_rst_r * y + ch * x + c] = saturate_cast<uchar>(v);
            }
    return rst;
}

template<class N>
Mat kernel_filter(Mat& src, void (*fill)(Mat&, int), const Mat& ker)
{
    Mat pad = padding(src, ker.rows / 2, fill);
    return apply_kernel<N>(pad, ker);
}

// 평균(박스) 필터 커널 생성 (문제 #2에서 사용)
Mat make_avg_ker(int p)
{
    int size = 2 * p + 1;
    double val = 1.0 / (size * size);
    return Mat(size, size, CV_64F, Scalar(val));
}

// 픽셀별 스칼라 함수 적용 헬퍼 (문제 #4에서 노이즈 추가에 활용)
template <typename F>
Mat pixelwise(Mat& src, F func)
{
    Mat rst(src.rows, src.cols, src.type());
    uchar* rst_P = rst.data;
    uchar* src_P = src.data;
    int l_rst_r = rst.step;
    int ch = src.channels();

    for (int y = 0; y < rst.rows; ++y)
        for (int x = 0; x < rst.cols; ++x)
            for (int c = 0; c < ch; ++c)
            {
                int idx = l_rst_r * y + ch * x + c;
                rst_P[idx] = saturate_cast<uchar>(func(src_P[idx]));
            }
    return rst;
}

// ============================
// 문제 #2: 평균 필터 크기 변화
// ============================
void P2(Mat& src, void (*fill)(Mat&, int))
{
    int cases[4] = { 1, 2, 5, 12 }; // 3x3, 5x5, 11x11, 25x25
    for (auto& x : cases)
    {
        Mat rst = kernel_filter<double>(src, fill, make_avg_ker(x));
        string name = "P2 Average Filter " + to_string(2 * x + 1) + "x" + to_string(2 * x + 1);
        imshow(name, rst);
    }
}

// ============================
// 문제 #3: OpenCV GaussianBlur 사용
// ============================
void P3(Mat& src)
{
    Mat rst;
    Size size_case[2] = { Size(3, 3), Size(7, 7) };
    double sigma_case[3] = { 0.1, 1.0, 5.0 };
    for (auto& ksz : size_case)
        for (auto& sig : sigma_case)
        {
            GaussianBlur(src, rst, ksz, sig);
            imshow("P3 GaussianBlur " + to_string(ksz.width) + "x" + to_string(ksz.height) + " sigma=" + to_string(sig), rst);
        }
}

// ============================
// 문제 #4: 가우시안 노이즈, 소금후추 노이즈 생성 + display
// ============================
// 가우시안 노이즈 추가 (픽셀별 독립 표본)
Mat add_gaussian_noise(Mat& src, mt19937& gen, normal_distribution<double>& g_dist)
{
    auto noise_add = [&gen, &g_dist](uchar v) {
        // v(0~255)에 정규분포 표본을 더하고 포화(saturate)
        double n = g_dist(gen);
        return v + n;
        };
    return pixelwise(src, noise_add);
}

// 소금/후추 노이즈 추가 (salt=255, pepper=0)
Mat add_salt_pepper_noise(Mat& src, mt19937& gen, uniform_real_distribution<double>& uni, double p_salt, double p_pepper)
{
    auto snp = [&gen, &uni, p_salt, p_pepper](uchar v) {
        double r = uni(gen);
        if (r < p_pepper) return (double) 0;            // pepper
        if (r > 1.0 - p_salt) return (double) 255;      // salt
        return (double) v;                               // 그대로 유지
        };
    return pixelwise(src, snp);
}

void P4(Mat& src, Mat& out_gauss, Mat& out_snp)
{
    // --- 사용자 입력 (기존 코드 보존) ---
    double g_m, g_s, s_p, p_p;
    cout << "Type Gaussian noise Parameter" << endl;
    cout << "mean : ";
    cin >> g_m;
    cout << "std : ";
    cin >> g_s;
    cout << "Type Salt&Pepper noise Parameter" << endl;
    cout << "Salt probability : ";
    cin >> s_p;
    cout << "Pepper probability : ";
    cin >> p_p;

    normal_distribution<double> g_dist(g_m, g_s);             // 정규분포 N(mean, std)
    uniform_real_distribution<double> snp_dist(0.0, 1.0);     // 균등분포 U(0,1)

    // --- 노이즈 영상 생성 ---
    out_gauss = add_gaussian_noise(src, gen, g_dist);
    out_snp = add_salt_pepper_noise(src, gen, snp_dist, s_p, p_p);

    // --- 결과 display ---
    imshow(string("P4 Gaussian Noise (mean=") + to_string(g_m) + ", std=" + to_string(g_s) + ")", out_gauss);
    imshow(string("P4 Salt&Pepper Noise (salt=") + to_string(s_p) + ", pepper=" + to_string(p_p) + ")", out_snp);

}

// ============================
// 문제 #5: Geometric mean filter (비선형)
// ============================
Mat geometric_mean_filter(Mat& src, int p, void (*fill)(Mat&, int))
{
    // 윈도 크기: (2p+1)^2
    int k = 2 * p + 1;
    int ch = src.channels();
    const double eps = 1e-8; // log(0) 방지용 미세값

    // 패딩: 경계 처리는 replicate가 보편적으로 자연스러움
    Mat pad = padding(src, p, fill);
    Mat dst(src.size(), src.type());

    uchar* P = pad.data;
    uchar* D = dst.data;
    int l_pad = pad.step;
    int l_dst = dst.step;

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            for (int c = 0; c < ch; ++c)
            {
                double sum_log = 0.0;
                for (int ky = -p; ky <= p; ++ky)
                    for (int kx = -p; kx <= p; ++kx)
                    {
                        int yy = y + p + ky;
                        int xx = (x + p + kx) * ch + c;
                        double v = static_cast<double>(P[yy * l_pad + xx]);
                        sum_log += log(v + eps);
                    }
                double gmean = exp(sum_log / (k * k));
                D[y * l_dst + x * ch + c] = saturate_cast<uchar>(gmean);
            }
        }
    }
    return dst;
}

// ============================
// 문제 #6: Median filter (nxn)
// ============================
Mat median_filter(Mat& src, int p, void (*fill)(Mat&, int))
{
    int k = 2 * p + 1;                  // 윈도 한 변 길이 (홀수)
    int ch = src.channels();

    Mat pad = padding(src, p, fill);
    Mat dst(src.size(), src.type());

    uchar* P = pad.data;
    uchar* D = dst.data;
    int l_pad = pad.step;
    int l_dst = dst.step;

    vector<uchar> buf;
    buf.reserve(k * k);

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            for (int c = 0; c < ch; ++c)
            {
                buf.clear();
                for (int ky = -p; ky <= p; ++ky)
                    for (int kx = -p; kx <= p; ++kx)
                    {
                        int yy = y + p + ky;
                        int xx = (x + p + kx) * ch + c;
                        buf.push_back(P[yy * l_pad + xx]);
                    }
                // 중앙값 추출 (k*k은 홀수)
                nth_element(buf.begin(), buf.begin() + buf.size() / 2, buf.end());
                uchar med = buf[buf.size() / 2];
                D[y * l_dst + x * ch + c] = med;
            }
        }
    }
    return dst;
}

// ============================
// 문제 #7: Adaptive median filter (Smax까지 창 크기 적응)
// ============================
static inline void window_stats(const Mat& pad, int cy, int cx, int r, int ch, int c, double& zmin, double& zmax, double& zmed)
{
    // pad는 이미 충분히 크게 패딩되어 있다고 가정
    vector<uchar> v;
    v.reserve((2 * r + 1) * (2 * r + 1));

    const int l = pad.step;
    const uchar* P = pad.data;

    zmin = 255.0; zmax = 0.0;
    for (int dy = -r; dy <= r; ++dy)
    {
        int yy = cy + dy;
        for (int dx = -r; dx <= r; ++dx)
        {
            int xx = (cx + dx) * ch + c;
            uchar val = P[yy * l + xx];
            v.push_back(val);
            if (val < zmin) zmin = val;
            if (val > zmax) zmax = val;
        }
    }
    nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    zmed = v[v.size() / 2];
}

Mat adaptive_median_filter(Mat& src, int Smax, void (*fill)(Mat&, int))
{
    // Smax는 홀수, 최소 3 가정. 반경으로는 pmax = (Smax-1)/2
    if (Smax % 2 == 0) Smax += 1;
    if (Smax < 3) Smax = 3;

    int pmax = (Smax - 1) / 2;
    int ch = src.channels();

    // 최대 반경으로 한 번에 패딩
    Mat pad = padding(src, pmax, fill);
    Mat dst(src.size(), src.type());

    uchar* D = dst.data;
    int l_dst = dst.step;

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            for (int c = 0; c < ch; ++c)
            {
                // Level A/B 절차
                int S = 3; // 시작 창 크기
                uchar out_val = 0;
                bool decided = false;

                while (true)
                {
                    int r = (S - 1) / 2;
                    int cy = y + pmax;        // pad 좌표계에서의 중심 y
                    int cx = x + pmax;        // pad 좌표계에서의 중심 x

                    double zmin, zmax, zmed;
                    window_stats(pad, cy, cx, r, ch, c, zmin, zmax, zmed);

                    double A1 = zmed - zmin;
                    double A2 = zmed - zmax;

                    if (A1 > 0 && A2 < 0)
                    {
                        // Level B
                        uchar zxy = pad.ptr<uchar>(cy)[cx * ch + c];
                        double B1 = zxy - zmin;
                        double B2 = zxy - zmax;
                        if (B1 > 0 && B2 < 0) out_val = zxy; // 그대로
                        else                     out_val = static_cast<uchar>(zmed); // 중앙값
                        decided = true;
                        break;
                    }
                    else
                    {
                        S += 2; // 창 확장
                        if (S > Smax)
                        {
                            out_val = static_cast<uchar>(zmed); // 최대치에서 중앙값
                            decided = true;
                            break;
                        }
                    }
                }
                D[y * l_dst + x * ch + c] = out_val;
            }
        }
    }
    return dst;
}

// ============================
// 문제 #8: PSNR 함수 + 결과 cout 으로 출력
// ============================
double psnr_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.size() == b.size() && a.type() == b.type());

    const int rows = a.rows;
    const int cols = a.cols;
    const int ch = a.channels();

    const uchar* A = a.data;
    const uchar* B = b.data;
    const int la = a.step;
    const int lb = b.step;

    double mse = 0.0;
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < ch; ++c)
            {
                int idxA = y * la + x * ch + c;
                int idxB = y * lb + x * ch + c;
                double diff = static_cast<double>(A[idxA]) - static_cast<double>(B[idxB]);
                mse += diff * diff;
            }
        }
    }
    mse /= (rows * cols * ch);
    if (mse <= 1e-12) return 99.0; // 매우 높은 PSNR로 간주
    double MAXI = 255.0;
    return 10.0 * log10((MAXI * MAXI) / mse);
}

// 간단한 텍스트로 PSNR 결과 표시
void psnr_cout(const vector<pair<string, double>>& psnrs)
{
    for (const auto& x : psnrs)
    {
        cout << x.first << " : " << x.second << endl;
    }
}
// ============================

// 문제 #9: 모든 조합에 대해 psnr구하고 cout으로 출력
void P9(Mat& src)
{
    cout << "\nP9 Results" << endl;
    vector<pair<string, double>> ps;
    // --- Gaussian noise + Mean filter (총 6조합) ---
    for (double s : {5.0, 10.0, 30.0})
    {
        normal_distribution<double> gdist(0.0, s);
        Mat g_noisy_p9 = add_gaussian_noise(src, gen, gdist);

        for (int p : {1, 2}) // 3x3, 5x5
        {
            Mat avgKer = make_avg_ker(p);
            Mat g_mean = kernel_filter<double>(g_noisy_p9, pad_replicate, avgKer);
            double val = psnr_u8(src, g_mean);
            ps.emplace_back("Gaussian(0," + to_string((int) s) + ") + Mean " + to_string(2 * p + 1) + "x" + to_string(2 * p + 1), val);
        }
    }


    // --- Salt & Pepper noise + Median / Adaptive Median (총 6조합) ---
    uniform_real_distribution<double> uni(0.0, 1.0);

    for (double pr : {0.05, 0.10, 0.25}) // 5%, 10%, 25%
    {
        Mat sp_noisy_p9 = add_salt_pepper_noise(src, gen, uni, pr, pr);

        Mat med3 = median_filter(sp_noisy_p9, 1, pad_replicate);           // 3x3
        Mat adap7 = adaptive_median_filter(sp_noisy_p9, 7, pad_replicate);   // Smax=7

        ps.emplace_back("S&P(" + to_string((int) (pr * 100)) + "%) + Median 3x3", psnr_u8(src, med3));
        ps.emplace_back("S&P(" + to_string((int) (pr * 100)) + "%) + Adaptive 7x7", psnr_u8(src, adap7));
    }

    psnr_cout(ps);
}


// 메인
// ============================
int main()
{
    Mat src = imread(src_path, IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "이미지를 열 수 없습니다: " << src_path << endl;
        return -1;
    }
    imshow("Original", src);

    // 문제 #2, #3 (기존)
    P2(src, pad_zero);
    P3(src);

    // 문제 #4: 노이즈 생성 (사용자 입력 포함), display
    Mat g_noisy, sp_noisy;
    P4(src, g_noisy, sp_noisy);

    // 문제 #5: Geometric mean filter (3x3, replicate 경계)
    Mat gmf_g = geometric_mean_filter(g_noisy, 1, pad_replicate);
    Mat gmf_sp = geometric_mean_filter(sp_noisy, 1, pad_replicate);
    imshow("P5 Geometric Mean 3x3 on Gaussian", gmf_g);
    imshow("P5 Geometric Mean 3x3 on S&P", gmf_sp);

    // 문제 #6: Median filter (3x3, replicate 경계)
    Mat med_g = median_filter(g_noisy, 1, pad_replicate);
    Mat med_sp = median_filter(sp_noisy, 1, pad_replicate);
    imshow("P6 Median 3x3 on Gaussian", med_g);
    imshow("P6 Median 3x3 on S&P", med_sp);

    // 문제 #7: Adaptive median filter (Smax=7, replicate 경계)
    Mat adap_g = adaptive_median_filter(g_noisy, 7, pad_replicate);
    Mat adap_sp = adaptive_median_filter(sp_noisy, 7, pad_replicate);
    imshow("P7 Adaptive Median Smax=7 on Gaussian", adap_g);
    imshow("P7 Adaptive Median Smax=7 on S&P", adap_sp);

    // 문제 #9: 모든 조합에 대해 psnr구하고 cout으로 출력
    P9(src);

    waitKey(0);
    return 0;
}
