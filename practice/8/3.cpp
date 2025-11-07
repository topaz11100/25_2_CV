// 3.cpp — HoughLinesP 실험 자동 실행 (CLI 인수 제거)
// Build: g++ -std=c++17 3.cpp -o houghp `pkg-config --cflags --libs opencv4`
// Run:   ./houghp   (입력/출력 경로는 아래 상수 수정)
// 설명:  보고서에 필요한 파라미터 변화 실험을 "상수 하드코딩"으로 모두 수행한다.
//        각 실험은 Canny 에지와 HoughLinesP 결과 이미지를 동시에 화면 표시(imshow) 및 파일 저장(imwrite)한다.
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;

// ====== 사용자 편집 상수 ======
// 입력 이미지 경로(실험용 원본). 필요시 보고서용 데이터로 바꿔서 재컴파일.
static const string kInputPath = "lena.png";
// 출력 폴더
static const string kOutDir = "실행결과/";
// 창 표시 여부(Headless 환경이면 false)
static const bool   kShowWindows = true;

// ====== 실험 파라미터 구조 ======
struct Exp {
    double rho;
    double theta_deg; // degree
    int    threshold;
    double minLineLength;
    double maxLineGap;
    double canny1;
    double canny2;
    int    aperture;
    string tag;       // 파일명/윈도우명에 사용
};

// 후처리: 라인 그리기
static void drawLines(Mat& vis, const vector<Vec4i>& linesP, const Scalar& color = Scalar(0, 0, 255), int thickness = 2)
{
    for (const auto& L : linesP)
    {
        line(vis, Point(L[0], L[1]), Point(L[2], L[3]), color, thickness, LINE_AA);
    }
}

// 한 번의 실험 실행
static void run_one(const Mat& src, const Exp& e)
{
// 1) 에지
    Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat edges; Canny(gray, edges, e.canny1, e.canny2, e.aperture, true);

    // 2) HoughLinesP
    vector<Vec4i> linesP;
    double theta = e.theta_deg * CV_PI / 180.0;
    HoughLinesP(edges, linesP, e.rho, theta, e.threshold, e.minLineLength, e.maxLineGap);

    // 3) 시각화
    Mat vis = src.clone();
    drawLines(vis, linesP);

    // 4) 저장/표시
    std::filesystem::create_directories(kOutDir);
    string pe = kOutDir + "/" + e.tag + "_edges.png";
    string pl = kOutDir + "/" + e.tag + "_lines.png";
    imwrite(pe, edges);
    imwrite(pl, vis);

    if (kShowWindows)
    {
        imshow(e.tag + " - edges", edges);
        imshow(e.tag + " - lines", vis);
        // 짧은 지연 (키 입력 시 다음으로)
        int key = waitKey(300);
        if (key >= 0) waitKey(1); // 키가 눌렸으면 잠깐 비움
    }

    cout << "[Saved] " << pe << ", " << pl << "  (detected=" << linesP.size() << " segments)\n";
}

// ====== 메인: 보고서용 실험계획 일괄 수행 ======
int main()
{
    Mat src = imread(kInputPath, IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "입력 이미지를 읽지 못했습니다: " << kInputPath << "\n";
        cerr << "kInputPath 상수를 수정하세요.\n";
        return 1;
    }

    // --- ① threshold 스윕 (다른 파라미터 고정) ---
    {
        double rho = 1.0, theta_deg = 1.0, minL = 30.0, gap = 10.0, c1 = 50.0, c2 = 150.0; int ap = 3;
        vector<int> Ts = { 30, 50, 80, 120 };
        for (int T : Ts)
        {
            run_one(src, Exp{ rho, theta_deg, T, minL, gap, c1, c2, ap, "sweep_T" + to_string(T) });
        }
    }

    // --- ② minLineLength 스윕 ---
    {
        double rho = 1.0, theta_deg = 1.0; int T = 50; double gap = 10.0, c1 = 50.0, c2 = 150.0; int ap = 3;
        vector<int> Ls = { 10, 30, 60, 100 };
        for (int L : Ls)
        {
            run_one(src, Exp{ rho, theta_deg, T, (double) L, gap, c1, c2, ap, "sweep_minL" + to_string(L) });
        }
    }

    // --- ③ maxLineGap 스윕 ---
    {
        double rho = 1.0, theta_deg = 1.0; int T = 80; double minL = 40.0, c1 = 50.0, c2 = 150.0; int ap = 3;
        vector<int> Gs = { 0, 5, 10, 20 };
        for (int G : Gs)
        {
            run_one(src, Exp{ rho, theta_deg, T, minL, (double) G, c1, c2, ap, "sweep_gap" + to_string(G) });
        }
    }

    // --- ④ 약한 Line 강조 프리셋 (Weak) ---
    {
        run_one(src, Exp{ 1.0, 1.0, 20, 10.0, 20.0, 30.0, 90.0, 3, "preset_weak" });
    }

    // --- ⑤ 강한 Line 위주 프리셋 (Strong) ---
    {
        run_one(src, Exp{ 1.0, 1.0, 120, 80.0, 5.0, 100.0, 200.0, 3, "preset_strong" });
    }

    if (kShowWindows)
    {
        cout << "모든 창을 닫으려면 아무 키나 누르세요.\n";
        waitKey(0);
        destroyAllWindows();
    }
    return 0;
}
