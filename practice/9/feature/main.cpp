// Feature Detection 실습 #1
// - 강의자료 Feature Detection 슬라이드 기반
// - 참고 코드(SURF) -> SIFT로 수정
// - 단일 이미지에서 SIFT feature point 검출
// - 파라미터 튜닝 및 블록 기반 검출 코드 포함

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 전역 SIFT 파라미터 (실습 28쪽: feature point 개수 조절용)
const int    SIFT_N_FEATURES = 0;      // 0: 제한 없음, 양수: 최대 feature 개수
const int    SIFT_N_OCTAVE_LAYERS = 3;
const double SIFT_CONTRAST_THRESH = 0.02;   // 기본값 0.04 보다 작게 주면 더 많은 특징점
const double SIFT_EDGE_THRESH = 10.0;
const double SIFT_SIGMA = 1.6;

// 블록 기반 검출용 파라미터 (실습 29쪽)
const int GRID_ROWS = 4; // M
const int GRID_COLS = 4; // N
const int GRID_TOTAL_FEATURES = 800;  // 전체 이미지에서 원하는 대략적인 feature 수

Ptr<SIFT> createSIFT(int nFeatures)
{
    return SIFT::create(
        nFeatures,
        SIFT_N_OCTAVE_LAYERS,
        SIFT_CONTRAST_THRESH,
        SIFT_EDGE_THRESH,
        SIFT_SIGMA
    );
}

// 전체 이미지에서 SIFT 특징점 검출
void detectSIFTGlobal(const Mat& gray,
    vector<KeyPoint>& keypoints,
    Mat& descriptors)
{
    Ptr<SIFT> sift = createSIFT(SIFT_N_FEATURES);
    sift->detectAndCompute(gray, noArray(), keypoints, descriptors);
}

// M x N 블록으로 나누어 블록별로 SIFT 특징점 검출
void detectSIFTGrid(const Mat& gray,
    vector<KeyPoint>& keypoints)
{
    keypoints.clear();

    int blockWidth = gray.cols / GRID_COLS;
    int blockHeight = gray.rows / GRID_ROWS;
    int nBlocks = GRID_ROWS * GRID_COLS;

    // 블록당 최대 feature 개수
    int nFeaturesPerBlock = GRID_TOTAL_FEATURES / max(1, nBlocks);

    Ptr<SIFT> sift = createSIFT(nFeaturesPerBlock);

    for (int by = 0; by < GRID_ROWS; ++by)
    {
        for (int bx = 0; bx < GRID_COLS; ++bx)
        {
            int x = bx * blockWidth;
            int y = by * blockHeight;

            // 마지막 블록에는 남은 픽셀을 모두 포함
            int w = (bx == GRID_COLS - 1) ? (gray.cols - x) : blockWidth;
            int h = (by == GRID_ROWS - 1) ? (gray.rows - y) : blockHeight;

            Rect roi(x, y, w, h);
            Mat block = gray(roi);

            vector<KeyPoint> localKeypoints;
            Mat descriptors;
            sift->detectAndCompute(block, noArray(), localKeypoints, descriptors);

            // 블록 좌상단 좌표만큼 keypoint 위치 보정
            for (auto& kp : localKeypoints)
            {
                kp.pt.x += static_cast<float>(x);
                kp.pt.y += static_cast<float>(y);
                keypoints.push_back(kp);
            }
        }
    }
}

int main()
{
    string imgPath = "/home/yongokhan/바탕화면/25_2_CV/source/Lena512.jpg";

    Mat imgColor = imread(imgPath, IMREAD_COLOR);
    if (imgColor.empty())
    {
        cerr << "Error: cannot read image: " << imgPath << endl;
        return -1;
    }

    Mat imgGray;
    cvtColor(imgColor, imgGray, COLOR_BGR2GRAY);

    // 1) 전체 이미지에서 SIFT 특징점 검출
    vector<KeyPoint> keypointsGlobal;
    Mat descriptorsGlobal;
    detectSIFTGlobal(imgGray, keypointsGlobal, descriptorsGlobal);

    Mat imgGlobal;
    drawKeypoints(
        imgColor,
        keypointsGlobal,
        imgGlobal,
        Scalar(0, 255, 0),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cout << "[Global] number of keypoints = "
        << keypointsGlobal.size() << endl;

   // 2) M x N 블록 기반 SIFT 특징점 검출
    vector<KeyPoint> keypointsGrid;
    detectSIFTGrid(imgGray, keypointsGrid);

    Mat imgGrid;
    drawKeypoints(
        imgColor,
        keypointsGrid,
        imgGrid,
        Scalar(255, 0, 0),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    cout << "[Grid]   number of keypoints = "
        << keypointsGrid.size() << endl;

   // 결과 영상 출력 및 저장
    imshow("SIFT Global", imgGlobal);
    imshow("SIFT Grid-based", imgGrid);

    imwrite("sift_global.png", imgGlobal);
    imwrite("sift_grid.png", imgGrid);

    cout << "Results saved as sift_global.png and sift_grid.png" << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}