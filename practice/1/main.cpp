#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

struct App
{
    VideoCapture cap;
    string winName = "Video Player";
    int totalFrames = 0;
    int curFrameIdx = 0;
    bool playing = false;     
    const int targetFPS = 30; 
    bool drawing = false;     
    Point ptStart, ptCur; 
    Mat curFrame;         

    
    map<int, vector<Rect>> boxes;

    static void onMouse(int event, int x, int y, int, void *userdata)
    {
        App *self = static_cast<App *>(userdata);
        if (!self)
            return;

        switch (event)
        {
        case EVENT_LBUTTONDOWN:
            self->drawing = true;
            self->ptStart = self->ptCur = Point(x, y);
            break;
        case EVENT_MOUSEMOVE:
            if (self->drawing)
                self->ptCur = Point(x, y);
            break;
        case EVENT_LBUTTONUP:
            if (self->drawing)
            {
                self->drawing = false;
                self->ptCur = Point(x, y);
                
                Rect r = self->normalizedRect(self->ptStart, self->ptCur);
                
                if (r.width > 1 && r.height > 1)
                {
                    
                    r &= Rect(0, 0, self->curFrame.cols, self->curFrame.rows);
                    if (r.width > 1 && r.height > 1)
                        self->boxes[self->curFrameIdx].push_back(r);
                }
            }
            break;
        default:
            break;
        }
        
        self->render();
    }

    Rect normalizedRect(const Point &a, const Point &b)
    {
        int x1 = min(a.x, b.x), y1 = min(a.y, b.y);
        int x2 = max(a.x, b.x), y2 = max(a.y, b.y);
        return Rect(x1, y1, x2 - x1, y2 - y1);
    }

    bool gotoFrame(int idx)
    {
        if (idx < 0)
            idx = 0;
        if (idx >= totalFrames)
            idx = totalFrames - 1;
        if (idx == curFrameIdx && !curFrame.empty())
            return true;

        cap.set(CAP_PROP_POS_FRAMES, idx);
        Mat f;
        if (!cap.read(f) || f.empty())
            return false;

        curFrame = f;
        curFrameIdx = idx;
        return true;
    }

    
    void render()
    {
        if (curFrame.empty())
            return;
        Mat vis = curFrame.clone();

        auto it = boxes.find(curFrameIdx);
        if (it != boxes.end())
        {
            for (const auto &r : it->second)
                rectangle(vis, r, Scalar(0, 0, 255), 2); 
        }
        if (drawing)
        {
            Rect r = normalizedRect(ptStart, ptCur);
            r &= Rect(0, 0, vis.cols, vis.rows);
            if (r.width > 1 && r.height > 1)
                rectangle(vis, r, Scalar(0, 0, 255), 2, LINE_8);
        }
        imshow(winName, vis);
    }
};

int main()
{
    
    const string VIDEO_PATH = "0.avi"; 
    
    App app;
    app.cap.open(VIDEO_PATH);
    if (!app.cap.isOpened())
    {
        cerr << "Failed to open: " << VIDEO_PATH << endl;
        return 1;
    }

    app.totalFrames = static_cast<int>(app.cap.get(CAP_PROP_FRAME_COUNT));
    if (app.totalFrames <= 0)
    {
        cerr << "Could not get total frame count.\n";
    }

    
    if (!app.gotoFrame(0))
    {
        cerr << "Failed to read first frame.\n";
        return 1;
    }

    namedWindow(app.winName, WINDOW_AUTOSIZE); 
    setMouseCallback(app.winName, App::onMouse, &app);
    app.render();

    const int delayMs = 1000 / app.targetFPS;

    cout << "Controls:\n"
         << "  Space : Play/Pause (30 FPS)\n"
         << "  N/n   : Next frame\n"
         << "  P/p   : Previous frame\n"
         << "  Esc/Q : Quit\n"
         << "Mouse:\n"
         << "  Left-drag to draw a red rectangle (stored per frame in memory)\n";

    while (true)
    {
        int key = waitKey(app.playing ? delayMs : 0);

        if (key == 27 || key == 'q' || key == 'Q')
        { 
            break;
        }
        else if (key == ' ')
        { 
            app.playing = !app.playing;
        }
        else if (key == 'n' || key == 'N')
        { 
            app.playing = false;
            if (app.gotoFrame(app.curFrameIdx + 1))
                app.render();
        }
        else if (key == 'p' || key == 'P')
        { 
            app.playing = false;
            if (app.gotoFrame(app.curFrameIdx - 1))
                app.render();
        }
  
        if (app.playing)
        {
            if (app.curFrameIdx + 1 < app.totalFrames)
            {
                if (app.gotoFrame(app.curFrameIdx + 1))
                    app.render();
            }
            else
            {
                app.playing = false; 
            }
        }
    }

    return 0;
}
