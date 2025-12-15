// Image Transformation & Video Stabilization - Practice #2
// - Based on feature_detection_실습_1.cpp (Motion Estimation + simple warping)
// - Practice #2 adds Path Planning (Gaussian smoothing of camera motion)
// - Motion Estimation: SIFT + Homography
// - Path Planning: Gaussian smoothing of homography sequence
// - Warping: Compensation using smoothed homographies (COLOR frames)
// - Video path is hard-coded as a constant.

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 하드코딩된 입력 영상 경로
const string VIDEO_PATH = "0.avi";   // 필요에 따라 파일명만 수정해서 사용

// Path Planning용 Gaussian Blur 파라미터 (실습2)
const int   GAUSSIAN_KERNEL_SIZE = 21;   // 시간축 방향 커널 크기 (홀수)
const double GAUSSIAN_SIGMA = 5.0;  // sigma 값이 클수록 더 부드럽게

// 두 프레임(그레이스케일) 사이의 Homography 추정: Feature + RANSAC
Mat estimateHomography(const Mat& srcGray, const Mat& dstGray)
{
    // SIFT 특징점 추출기 생성
    int nFeatures = 400; // 대략적인 최대 특징점 수
    Ptr<SIFT> detector = SIFT::create(nFeatures);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(srcGray, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(dstGray, noArray(), keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty())
    {
        // 특징점이 거의 없는 경우: 항등 변환으로 처리
        return Mat::eye(3, 3, CV_64F);
    }

    // FLANN 기반 매칭 (SIFT는 float descriptor 이므로 NORM_L2 사용)
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.7f; // Lowe의 ratio test
    vector<Point2f> srcPoints;
    vector<Point2f> dstPoints;

    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i].size() < 2) continue;

        const DMatch& m1 = knn_matches[i][0];
        const DMatch& m2 = knn_matches[i][1];

        if (m1.distance < ratio_thresh * m2.distance)
        {
            srcPoints.push_back(keypoints1[m1.queryIdx].pt);
            dstPoints.push_back(keypoints2[m1.trainIdx].pt);
        }
    }

    if (srcPoints.size() < 4 || dstPoints.size() < 4)
    {
        // Homography 계산에 최소 4쌍 이상 필요
        return Mat::eye(3, 3, CV_64F);
    }

    Mat H = findHomography(srcPoints, dstPoints, RANSAC);

    if (H.empty())
        H = Mat::eye(3, 3, CV_64F);

    H.convertTo(H, CV_64F);
    return H;
}

// 프레임 시퀀스에서 카메라 궤적(누적 Homography) H_list[i] 생성
void buildCameraTrajectory(const vector<Mat>& framesGray, vector<Mat>& H_list)
{
    H_list.clear();
    if (framesGray.empty()) return;

    Mat cumulative = Mat::eye(3, 3, CV_64F); // 첫 프레임 기준 좌표계
    H_list.push_back(cumulative.clone());     // H_0 = I

    for (size_t i = 1; i < framesGray.size(); ++i)
    {
        Mat H = estimateHomography(framesGray[i - 1], framesGray[i]);
        // H_i * ... * H_1 형태로 누적 (카메라 Motion)
        cumulative = H * cumulative;
        H_list.push_back(cumulative.clone());
    }
}

// 실습2: Homography 시퀀스에 대한 Path Planning (Gaussian smoothing)
// 강의자료의 smoothHomographies 예시 코드와 동일한 로직
void smoothHomographies(vector<Mat>& Hs,
    int kernelSize = GAUSSIAN_KERNEL_SIZE,
    double sigma = GAUSSIAN_SIGMA)
{
    int n = static_cast<int>(Hs.size());
    if (n == 0) return;

    // Homography 8 DoF를 (n x 8) 행렬에 저장 (H(2,2)는 1로 고정)
    Mat data(n, 8, CV_64F);

    for (int i = 0; i < n; ++i)
    {
        Mat H = Hs[i];
        H.convertTo(H, CV_64F);

        data.at<double>(i, 0) = H.at<double>(0, 0);
        data.at<double>(i, 1) = H.at<double>(0, 1);
        data.at<double>(i, 2) = H.at<double>(0, 2);
        data.at<double>(i, 3) = H.at<double>(1, 0);
        data.at<double>(i, 4) = H.at<double>(1, 1);
        data.at<double>(i, 5) = H.at<double>(1, 2);
        data.at<double>(i, 6) = H.at<double>(2, 0);
        data.at<double>(i, 7) = H.at<double>(2, 1);
    }

    // 커널 크기는 홀수로 맞춤
    if (kernelSize % 2 == 0) kernelSize += 1;

    // 시간축 방향으로 Gaussian Blur 적용 (Path Planning: low-pass filtering)
    GaussianBlur(data, data, Size(1, kernelSize), sigma);

    // 다시 Homography 행렬로 복원
    for (int i = 0; i < n; ++i)
    {
        Mat H = Mat::eye(3, 3, CV_64F);
        H.at<double>(0, 0) = data.at<double>(i, 0);
        H.at<double>(0, 1) = data.at<double>(i, 1);
        H.at<double>(0, 2) = data.at<double>(i, 2);
        H.at<double>(1, 0) = data.at<double>(i, 3);
        H.at<double>(1, 1) = data.at<double>(i, 4);
        H.at<double>(1, 2) = data.at<double>(i, 5);
        H.at<double>(2, 0) = data.at<double>(i, 6);
        H.at<double>(2, 1) = data.at<double>(i, 7);
        // H(2,2)는 1로 유지

        Hs[i] = H;
    }
}

int main()
{
    // 1) 동영상 읽기
    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened())
    {
        cerr << "Error: cannot open video: " << VIDEO_PATH << endl;
        return -1;
    }

    vector<Mat> framesColor;
    vector<Mat> framesGray;

    Mat frame;
    while (cap.read(frame))
    {
        framesColor.push_back(frame.clone());

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        framesGray.push_back(gray);
    }

    if (framesGray.empty())
    {
        cerr << "Error: no frames read from video." << endl;
        return -1;
    }

    // 2) Motion Estimation + 카메라 궤적 계산 (실습1과 동일한 부분)
    vector<Mat> H_list;          // 원본 카메라 Motion (누적 Homography)
    buildCameraTrajectory(framesGray, H_list);

    // 3) Path Planning (실습2 추가 부분)
    //    - Homography 시퀀스를 Gaussian Blur로 부드럽게 만들어 Smooth Path 생성
    vector<Mat> H_smooth = H_list;  // 원본을 복사해서 smoothing
    smoothHomographies(H_smooth);

    // 4) Warping: Smooth Path와 원본 Path 차이를 보상하는 Homography 적용
    //    Compensation H_comp = H_smooth * H_original^{-1}

    Size frameSize = framesGray[0].size();
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // FPS 정보를 얻지 못하면 기본값 사용

    // 결과 동영상 저장 (컬러)
    VideoWriter writer("stabilized_practice2.avi",
        VideoWriter::fourcc('X', 'V', 'I', 'D'),
        fps,
        frameSize,
        true); // true: color

    if (!writer.isOpened())
    {
        cerr << "Warning: cannot open VideoWriter. Only showing result without saving." << endl;
    }

    for (size_t i = 0; i < framesGray.size(); ++i)
    {
        // 실습1: H_comp = H_list[i].inv();  (Smooth Path = Identity)
        // 실습2: H_comp = H_smooth[i] * H_list[i].inv();
        Mat H_comp = H_smooth[i] * H_list[i].inv();

        Mat stabilizedColor;
        warpPerspective(framesColor[i], stabilizedColor, H_comp, frameSize,
            INTER_LINEAR, BORDER_REPLICATE);

        if (writer.isOpened())
            writer.write(stabilizedColor);

        imshow("Original (Color)", framesColor[i]);
        imshow("Stabilized (Practice 2, Color)", stabilizedColor);

        char key = (char) waitKey(30);
        if (key == 27) break; // ESC 키로 종료
    }

    return 0;
}