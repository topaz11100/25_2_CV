#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

class fpstimer
{
public:
    fpstimer(int f) : fps(f), start_T(0) {}
    void start() { start_T = getTickCount(); }
    double end()
    {
        double s = (getTickCount() - start_T) / getTickFrequency();
        start_T = 0;
        return s;
    }
    void until_fps_delay(double elapsed)
    {
        double remain = (1.0 / fps) - elapsed;
        if (remain <= 0)
            return;
        this_thread::sleep_for(chrono::duration<double>(remain));
    }

private:
    int fps;
    int64 start_T;
};

class video_player
{
public:
    // 웹캠 금지: 문자열 경로만 받음
    video_player(const string &path, int target_fps = 30)
        : cap(path), timer(target_fps), cur_idx(0), playing(false), dragging(false)
    {
        if (!cap.isOpened())
        {
            cerr << "Failed to open: " << path << endl;
            exit(1);
        }
        total = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
        win = "Video Player";
        namedWindow(win);
        // 첫 프레임 로드 & 표시
        load_frame(0);
        display();
        // 마우스 콜백
        setMouseCallback(win, &video_player::mouse_thunk, this);
        loop();
    }

private:
    // ===== 메인 루프 =====
    void loop()
    {
        while (true)
        {
            timer.start();
            int key = pollKey();
            if (key == 27 || key == 'q' || key == 'Q')
                break;

            if (key == ' ')
                playing = !playing;

            if (key == 'p' || key == 'P')
            {
                playing = false;
                step(-1);
            }
            else if (key == 'n' || key == 'N')
            {
                playing = false;
                step(+1);
            }

            if (playing)
                step(+1);

            // 드래그 중이면 임시 오버레이 포함해서 계속 갱신
            display();

            timer.until_fps_delay(timer.end());
        }
    }

    // ===== 프레임 이동 =====
    void step(int delta)
    {
        int next = cur_idx + delta;
        if (next < 0)
            next = 0;
        if (total > 0 && next >= total)
        {
            next = max(0, total - 1);
            playing = false;
        }
        if (next != cur_idx)
            load_frame(next);
    }

    // 원본 프레임 로드(항상 유지)
    void load_frame(int index)
    {
        cap.set(CAP_PROP_POS_FRAMES, index);
        Mat f;
        if (!cap.read(f) || f.empty())
        {
            // 읽기 실패 시 현재 위치 유지
            return;
        }
        base = std::move(f);
        cur_idx = index;
    }

    // ===== 표시 로직 =====
    void display()
    {
        // annot 저장본 있으면 그걸 기본으로, 없으면 원본
        Mat canvas = has_annot(cur_idx) ? annot[cur_idx].clone() : base.clone();

        // 드래그 중이면 임시 사각형 오버레이(확정은 mouse up에서)
        if (dragging)
        {
            Rect r = norm_rect(start_pt, cur_pt, canvas.size());
            rectangle(canvas, r, Scalar(0, 0, 255), 2, LINE_8);
        }
        imshow(win, canvas);
    }

    // ===== 마우스 콜백 =====
    static void mouse_thunk(int event, int x, int y, int flags, void *userdata)
    {
        auto *self = reinterpret_cast<video_player *>(userdata);
        self->on_mouse(event, x, y, flags);
    }

    void on_mouse(int event, int x, int y, int /*flags*/)
    {
        switch (event)
        {
        case EVENT_LBUTTONDOWN:
            playing = false; // 편집 시작 시 일시정지
            dragging = true;
            start_pt = cur_pt = Point(x, y);
            display();
            break;
        case EVENT_MOUSEMOVE:
            if (dragging)
            {
                cur_pt = Point(x, y);
                display();
            }
            break;
        case EVENT_LBUTTONUP:
            if (dragging)
            {
                dragging = false;
                cur_pt = Point(x, y);
                commit_rect(); // 확정 → 맵에 저장
                display();
            }
            break;
        }
    }

    // 사각형 확정: annot 맵에 저장(프레임별 완성본 Mat 캐시)
    void commit_rect()
    {
        if (base.empty())
            return;
        Rect r = norm_rect(start_pt, cur_pt, base.size());
        if (r.width <= 0 || r.height <= 0)
            return;

        if (!has_annot(cur_idx))
        {
            annot[cur_idx] = base.clone(); // 최초 편집 시 원본 복사해 저장본 생성
        }
        rectangle(annot[cur_idx], r, Scalar(0, 0, 255), 2, LINE_8);
    }

    static Rect norm_rect(Point a, Point b, const Size &sz)
    {
        int x1 = std::clamp(min(a.x, b.x), 0, sz.width - 1);
        int y1 = std::clamp(min(a.y, b.y), 0, sz.height - 1);
        int x2 = std::clamp(max(a.x, b.x), 0, sz.width - 1);
        int y2 = std::clamp(max(a.y, b.y), 0, sz.height - 1);
        return Rect(Point(x1, y1), Point(x2, y2));
    }

    inline bool has_annot(int idx) const
    {
        return annot.find(idx) != annot.end();
    }

private:
    string win;
    VideoCapture cap;
    fpstimer timer;
    int total = 0;
    int cur_idx = 0;

    Mat base;                      // 현재 프레임의 원본
    unordered_map<int, Mat> annot; // 프레임 번호 → 편집본(Mat)

    // 드래그 상태
    bool playing;
    bool dragging;
    Point start_pt, cur_pt;
};

int main()
{
    string path = "0.avi";
    int target_fps = 30;

    video_player player(path, target_fps);
    return 0;
}
