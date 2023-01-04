#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <linux/types.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdio>
#include <string.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <errno.h>
#include <gst/gst.h>
#include <gst/gstminiobject.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <sys/shm.h>
#include <semaphore.h>
#include <chrono>
#include <cmath>
#include <termios.h>
#include "loopqueue.hpp"
#include "http_stream.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

#define WINDOW_NAME  "gst_camera"
#define DEFAULT_RTSP_PORT "8554"
#define HEIGHT 1080
#define WIDTH  1920
bool USE_GST_RTSP    = true; 
bool USE_GST_SHOW    = true;
bool USE_OPENCV_SHOW = false;

static char *port = (char *) DEFAULT_RTSP_PORT;
bool start_gst = false;
volatile int stop_rc = 0;

struct shared_use_st
{
    int Index;
    char Buffer[WIDTH*HEIGHT*3];
    sem_t sem;
};

void *shm = NULL;
struct shared_use_st *shared = NULL;
int shmid;

cv::Mat Image;
char BGR_IMG[WIDTH*HEIGHT*3] = {0};

typedef struct _CustomData {
    GstElement *pipeline, *appsrc, *parse, *decoder, *scale, *filter1, *conv1, *conv2, *encoder, *filter2, *sink, *mTextOverlay, *gdkpixbufoverlay;
    GMainLoop *loop;
    GstBus *bus;
    guint bus_watch_id;
    gboolean playing;  /* Playing or Paused */
    gdouble rate;      /* Current playback rate (can be negative) */
} CustomData;

CustomData data;

typedef struct Msg
{
    int len;
    char* data;
} msg;

LoopQueue<msg*> gstqueue(32);
LoopQueue<msg*> rtspqueue(32);

pthread_mutex_t gstMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  gstCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t rtspMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  rtspCond = PTHREAD_COND_INITIALIZER;
int count_t = 0;
bool recording = false;

static void signal_handler(int signo)
{
    printf("SIGINT: Closing accessory\n");
    stop_rc = 1;
    usleep(100000);
    exit(-1);
}

int frame_num = 0;
static void cb_need_data(GstElement *appsrc, guint size, gpointer user_data)
{
    GstBuffer *buf;
    static GstClockTime timestamp = 0;
    GstFlowReturn ret;
    GstMapInfo map;

    count_t++;
    pthread_mutex_lock(&gstMutex);
    while (0 == gstqueue.getSize())
    {
        pthread_cond_wait(&gstCond, &gstMutex);
        usleep(100);
    }
    buf = gst_buffer_new_allocate(NULL, gstqueue.top()->len, NULL);
    if (!buf){
        pthread_mutex_unlock(&gstMutex);
        return NULL;
    }

    if(!gst_buffer_map(buf, &map, GST_MAP_WRITE)){
        gst_buffer_unref(buf);
        pthread_mutex_unlock(&gstMutex);
        return NULL;
    }else{
        memcpy((guchar *)map.data, (guchar *)(gstqueue.top()->data), gstqueue.top()->len);
    }
    pthread_mutex_unlock(&gstMutex);

    GST_BUFFER_PTS(buf) = timestamp;
    GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(1, GST_MSECOND, 30);
    timestamp += GST_BUFFER_DURATION(buf);
    //add overlay
    frame_num++;
    std::string text_frame = to_string(frame_num);
    //g_object_set (G_OBJECT (data.mTextOverlay), "font_desc", "Ahafoni CLM Bold 20", NULL);
    //g_object_set (G_OBJECT (data.mTextOverlay), "font_desc", "Ubuntu Mono 12pt", NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "font_desc", "Sans 15", NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "xpad", 15, NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "ypad", 0, NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "wrap-mode", -1, NULL); // no wrapping
    g_object_set (G_OBJECT (data.mTextOverlay), "scale-mode", 2, NULL); // display
    g_object_set (G_OBJECT (data.mTextOverlay), "color", 0xFFFFFFE0, NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "valignment", 2,NULL);//up
    g_object_set (G_OBJECT (data.mTextOverlay), "halignment", 0,NULL);//left
    g_object_set (G_OBJECT (data.mTextOverlay), "shaded-background", FALSE, NULL);
    g_object_set (G_OBJECT (data.mTextOverlay), "text", text_frame.c_str(), NULL);

    //g_object_set (G_OBJECT (data.mTextOverlay), "valignment", 1,NULL);//down
    //g_object_set (G_OBJECT (data.mTextOverlay), "valignment", 2,NULL);//up
    //g_object_set (G_OBJECT (data.mTextOverlay), "valignment", 4,NULL);//center
    //g_object_set (G_OBJECT (data.mTextOverlay), "halignment", 0,NULL);//left
    //g_object_set (G_OBJECT (data.mTextOverlay), "halignment", 1,NULL);//center
    //g_object_set (G_OBJECT (data.mTextOverlay), "halignment", 2,NULL);//right
    //g_object_set (data.mTextOverlay, "text", text_frame.c_str(), NULL);

    //add pixbufoverlay
    //g_object_set(data.gdkpixbufoverlay, "location", "/home/nvidia/Pictures/test.png", NULL);
    //g_object_set(data.gdkpixbufoverlay, "offset-x", 100, NULL);
    //g_object_set(data.gdkpixbufoverlay, "offset-y", 100, NULL);

    g_signal_emit_by_name(appsrc, "push-buffer", buf, &ret);

    if(buf){
        gst_buffer_unmap (buf, &map);
        gst_buffer_unref(buf);
    }
    if (ret != 0)
    {
        //g_main_loop_quit(loop);
    }
    //printf("getqueue.getSize() = %d\n", gstqueue.getSize());
    delete gstqueue.top()->data;
    delete gstqueue.top();
    gstqueue.pop();
}

/* Process keyboard input */
static gboolean handle_keyboard (GIOChannel *source, GIOCondition cond, CustomData *data) {
    //printf("handle_keyboard!\n");
    gchar *str = NULL;
    if(g_io_channel_read_line (source, &str, NULL, NULL, NULL) != G_IO_STATUS_NORMAL) {
        return TRUE;
    }
    //g_print("%s\n",str);
    switch(g_ascii_tolower (str[0])) {
        case 'p':
            //data->playing = !data->playing;
            //gst_element_set_state (data->pipeline, data->playing ? GST_STATE_PLAYING : GST_STATE_PAUSED);
            data->playing = false;
            gst_element_set_state (data->pipeline, GST_STATE_PAUSED);
            g_print ("Setting state to %s\n", data->playing ? "PLAYING" : "PAUSE");
            break;
        case 'r':
            data->playing = true;
            gst_element_set_state (data->pipeline, GST_STATE_PLAYING);
            g_print ("Setting state to %s\n", data->playing ? "PLAYING" : "PAUSE");
            break;
        case 'q':
            printf("g_main_loop_quit!\n");
            g_main_loop_quit (data->loop);
            stop_rc = 1;
            usleep(10000);
            exit(-1);
            break;
        default:
            break;
    }
    g_free (str);
    return TRUE;
}

void *gst_show(void *arg)
{
    while(!start_gst)
    {
        usleep(100);
    }
    // init GStreamer
    //CustomData data;
    GIOChannel *io_stdin;
    gst_init(NULL, NULL);
    memset(&data, 0, sizeof(data));
    data.loop = g_main_loop_new(NULL, FALSE);
    //user keypress
    io_stdin = g_io_channel_unix_new (fileno (stdin));
    g_io_add_watch (io_stdin, G_IO_IN, (GIOFunc)handle_keyboard, &data); ///watches the keyboard interrupt

    // nv3dsink nveglglessink
    // setup pipeline
    data.pipeline = gst_pipeline_new("pipeline");
    data.appsrc   = gst_element_factory_make("appsrc", "videosrc");
    data.conv1    = gst_element_factory_make ("videoconvert", "conv1");
    data.filter1  = gst_element_factory_make ("capsfilter", "filter1");
    //data.conv2    = gst_element_factory_make ("nvvidconv", "conv2");
    data.mTextOverlay = gst_element_factory_make ("textoverlay", "textoverlay");
    data.gdkpixbufoverlay = gst_element_factory_make ("gdkpixbufoverlay", "overlaytool");
    data.sink     = gst_element_factory_make ("nv3dsink", "sink");
    //data.sink     = gst_element_factory_make ("nvdrmvideosink", "sink");

    if (!data.pipeline || !data.appsrc || !data.conv1 || !data.filter1 || !data.mTextOverlay || !data.gdkpixbufoverlay || !data.sink) {
        g_printerr("One element could not be created.\n");
    }

    g_object_set (G_OBJECT (data.appsrc),
        "stream-type" , 0 ,
        "format" , GST_FORMAT_TIME , NULL);

    g_object_set (G_OBJECT (data.appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, WIDTH,
            "height", G_TYPE_INT, HEIGHT,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL), NULL);

    g_object_set (G_OBJECT (data.filter1), "caps",
        gst_caps_new_simple ("video/x-raw(memory:NVMM)",
            "format", G_TYPE_STRING, "BGRx", NULL), NULL);

    g_object_set(data.sink, "sync", false, NULL);
    g_object_set(data.sink, "async", false, NULL);

    //nv3dsink -e
    gst_bin_add_many(GST_BIN(data.pipeline), data.appsrc, data.conv1, data.filter1, data.mTextOverlay, data.gdkpixbufoverlay, data.sink, NULL);
    gst_element_link_many(data.appsrc, data.conv1, data.filter1, data.mTextOverlay, data.gdkpixbufoverlay, data.sink, NULL);
    //gst_bin_add_many(GST_BIN(data.pipeline), data.appsrc, data.conv1, data.filter1, data.sink, NULL);
    //gst_element_link_many(data.appsrc, data.conv1, data.filter1, data.sink, NULL);

    g_signal_connect(data.appsrc, "need-data", G_CALLBACK(cb_need_data), NULL);

    /* play */
    std::cout << "start sender pipeline" << std::endl;
    gst_element_set_state(data.pipeline, GST_STATE_PLAYING);

    g_main_loop_run(data.loop);

    /* clean up */
    gst_element_set_state(data.pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(data.pipeline));
    g_main_loop_unref(data.loop);
}

//ffmpeg  -i rtsp://192.168.1.6:8554/test -vcodec  copy  -t 60  -y test.mp4
typedef struct
{
    gboolean white;
    GstClockTime timestamp;
} MyContext;

/* called when we need to give data to appsrc */
static void need_data (GstElement * appsrc, guint unused, MyContext * ctx)
{
    GstBuffer *buf;
    guint buffersize;
    GstFlowReturn ret;
    GstMapInfo map;
    GstClockTime pts, dts;
    count_t++;

    pthread_mutex_lock(&rtspMutex);
    while (0 == rtspqueue.getSize())
    {
        pthread_cond_wait(&rtspCond, &rtspMutex);
        usleep(100);
    }
    buf = gst_buffer_new_allocate(NULL, rtspqueue.top()->len, NULL);
    if (!buf){
        pthread_mutex_unlock(&rtspMutex);
        return NULL;
    }
    if(!gst_buffer_map(buf, &map, GST_MAP_WRITE)){
        gst_buffer_unref(buf);
        pthread_mutex_unlock(&rtspMutex);
        return NULL;
    }else{
        memcpy((guchar *)map.data, (guchar *)(rtspqueue.top()->data), rtspqueue.top()->len);
    }
    pthread_mutex_unlock(&rtspMutex);

    GST_BUFFER_PTS(buf) = ctx->timestamp;
    GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(1, GST_SECOND, 30);
    ctx->timestamp += GST_BUFFER_DURATION(buf);
    g_signal_emit_by_name(appsrc, "push-buffer", buf, &ret);
    if(buf){
        gst_buffer_unmap (buf, &map);
        gst_buffer_unref(buf);
    }
    if (ret != 0)
    {
        printf("g_main_loop_quit\n");
    }
    delete rtspqueue.top()->data;
    delete rtspqueue.top();
    rtspqueue.pop();
}

static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media, gpointer user_data)
{
    printf("media_configure\n");
    GstElement *element, *appsrc;
    MyContext *ctx;

    /* get the element used for providing the streams of the media */
    element = gst_rtsp_media_get_element (media);

    /* get our appsrc, we named it 'videosrc' with the name property */
    appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "videosrc");
    g_object_set (G_OBJECT (appsrc), 
        "stream-type" , 0 , //rtsp
        "format" , GST_FORMAT_TIME , NULL);

    g_object_set (G_OBJECT (appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, WIDTH,
            "height", G_TYPE_INT, HEIGHT,
            "framerate", GST_TYPE_FRACTION, 30, 1,
            "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL), NULL);

    ctx = g_new0 (MyContext, 1);
    ctx->white = FALSE;
    ctx->timestamp = 0;
    /* make sure ther datais freed when the media is gone */
    g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
        (GDestroyNotify) g_free);

    /* install the callback that will be called when a buffer is needed */
    g_signal_connect (appsrc, "need-data", (GCallback) need_data, ctx);
    gst_object_unref (appsrc);
    gst_object_unref (element);
}

void* multirtsp(void *args)
{
    GMainLoop *loop = (GMainLoop *) args;
    g_main_loop_run (loop);
}

void *gst_rtsp(void *arg)
{
    while(!start_gst)
    {
        usleep(100);
    }
    GMainLoop *loop;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;

    gst_init (NULL, NULL);
    loop = g_main_loop_new (NULL, FALSE);

    pthread_t m_rtsp;
    pthread_create(&m_rtsp, NULL, multirtsp, loop);
    server = gst_rtsp_server_new ();
    g_object_set (server, "service", port, NULL); 
    
    mounts = gst_rtsp_server_get_mount_points (server);
    factory = gst_rtsp_media_factory_new ();

    //SW encode
    //gst_rtsp_media_factory_set_launch (factory,
    //        "( appsrc name=videosrc is-live=true ! videoconvert ! video/x-raw, format=I420 ! x264enc bitrate=4000000 ! rtph264pay config-interval=10 name=pay0 pt=96 )");

    //HW encode
    //Profile Description 0 Baseline profile 2 Main profile 4 High profile
    //Hardware Preset Level 0 DisablePreset 1 UltraFastPreset 2 FastPreset
    gst_rtsp_media_factory_set_launch (factory,
            "( appsrc name=videosrc is-live=true ! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h264enc maxperf-enable=1 control-rate=0 bitrate=4000000 preset-level=2 profile=2 ! rtph264pay name=pay0 pt=96 sync=false )");

    gst_rtsp_media_factory_set_shared (factory, TRUE); 
    g_signal_connect (factory, "media-configure", (GCallback) media_configure, NULL);
    gst_rtsp_mount_points_add_factory (mounts, "/test", factory);
    g_object_unref (mounts);
    gst_rtsp_server_attach (server, NULL);
    /* start serving */
    g_print ("stream ready at rtsp://127.0.0.1:8554/test\n");
    pthread_join(m_rtsp, NULL);
}

int shm_init(void)
{
    int i = 0;
    shmid = shmget((key_t)123, sizeof(struct shared_use_st), 0666|IPC_CREAT);
    if(shmid == -1)
    {
        exit(EXIT_FAILURE);
    }
    shm = shmat(shmid, (void*)0, 0);
    if(shm == (void*)-1)
    {
        exit(EXIT_FAILURE);
    }
    printf("Memory attached at %ld\n", (intptr_t)shm);
    shared = (struct shared_use_st*)shm;
    shared->Index = 0;
    sem_init(&(shared->sem),1,1);
    return 1;
}

void *captureImage(void *arg)
{
    cv::Mat img_input;
    unsigned int capwidth = 0;
    unsigned int capheight = 0;
    unsigned int framerate = 0;
    unsigned int frame_num = 0;
    cv::VideoCapture capture;
    //camera source support: usb camera, csi, rtsp
    //usb cam: MJPG
    capture.open("v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv flip-method=0 ! video/x-raw,width=1920,height=1080,format=BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    //usb cam: UYVY
    //capture.open("v4l2src device=/dev/video0 ! video/x-raw, width=1920, height=1080, format=UYVY, framerate=30/1 ! videoconvert ! video/x-raw, format=(string)BGR, width=1920,height=1080 ! appsink max-buffers=1 drop=false sync=false", cv::CAP_GSTREAMER);

    //csi
    //capture.open("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080 ! nvvidconv flip-method=2 ! video/x-raw,width=1920,height=1080,format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    //capture.open("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080, format=(string)NV12 ! nvvidconv flip-method=2 ! video/x-raw,width=1920,height=1080,format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    //rtsp
    //capture.open("rtspsrc location=rtsp://admin:hyzn1234@192.168.1.64:554/h264/ch1/main/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink sync=false", cv::CAP_GSTREAMER);

    if(!capture.isOpened()){
        std::cout<<"VideoCapture or VideoWriter not opened"<<std::endl;
        exit(-1);
    }
    std::cout<<"VideoCapture or VideoWriter opened"<<std::endl;

    int key = -1;
    if(USE_OPENCV_SHOW){
        cv::namedWindow(WINDOW_NAME,CV_WINDOW_NORMAL);
        cv::resizeWindow(WINDOW_NAME,960,540);
        cv::moveWindow(WINDOW_NAME, 0, 0);
        cv::startWindowThread();
    }
    capwidth = capture.get(CAP_PROP_FRAME_WIDTH);
    capheight = capture.get(CAP_PROP_FRAME_HEIGHT);
    framerate = capture.get(CAP_PROP_FPS);
    std::cout << "width : " << capwidth << " height : " << capheight << " framerate : " << framerate << std::endl;
    char *image;

    auto start = std::chrono::system_clock::now();
    while(!stop_rc)
    {
        capture >> img_input;
        start_gst = true;
        //auto start = std::chrono::system_clock::now();
        if(USE_OPENCV_SHOW){
            cv::startWindowThread();
            cv::imshow(WINDOW_NAME, img_input);
            key = cv::waitKey(10);
            if (key == 'q'){
                break;
            }
        }
        //process image
        image = (char*)img_input.data;
        //save image
        if(sem_wait(&(shared->sem)) == -1)
        {
            printf("P ERROR!\n");
            exit(EXIT_FAILURE);
        }
        memcpy(shared->Buffer, image, WIDTH*HEIGHT*3);
        sem_post(&shared->sem);
        if(USE_GST_RTSP){
            if(rtspqueue.getSize() < 5){
                msg *mp_rtsp = new msg;
                mp_rtsp->len = WIDTH*HEIGHT*3;
                mp_rtsp->data = new uint8_t[WIDTH*HEIGHT*3];
                memcpy(mp_rtsp->data, image, WIDTH*HEIGHT*3);
                pthread_mutex_lock(&rtspMutex);
                rtspqueue.push(mp_rtsp);
                pthread_mutex_unlock(&rtspMutex);
                pthread_cond_signal(&rtspCond);
            }
        }

        if(USE_GST_SHOW){
            msg *mp_gst = new msg;
            mp_gst->len = WIDTH*HEIGHT*3;
            mp_gst->data = new uint8_t[WIDTH*HEIGHT*3];
            memcpy(mp_gst->data, image, WIDTH*HEIGHT*3);
            pthread_mutex_lock(&gstMutex);
            gstqueue.push(mp_gst);
            pthread_mutex_unlock(&gstMutex);
            pthread_cond_signal(&gstCond);
        }
        auto end = std::chrono::system_clock::now();
        int fps = 1000.0/std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    if(USE_OPENCV_SHOW){
        capture.release();
        cv::destroyAllWindows();
    }
}

bool parse_args(int argc, char** argv) {
    if (argc > 2) return false;
    if (std::string(argv[1]) == "-g") {
        USE_GST_SHOW = true;
        USE_GST_RTSP = false;
        USE_OPENCV_SHOW = false;
    } else if(std::string(argv[1]) == "-r"){
        USE_GST_SHOW = false;
        USE_GST_RTSP = true;
        USE_OPENCV_SHOW = false;
    } else if(std::string(argv[1]) == "-gr"){
        USE_GST_SHOW = true;
        USE_GST_RTSP = true;
        USE_OPENCV_SHOW = false;
    } else if(std::string(argv[1]) == "-o"){
        USE_GST_SHOW = false;
        USE_GST_RTSP = false;
        USE_OPENCV_SHOW = true;
    } else{
        return false;
    }
    return true;
}        

int main(int argc, char** argv)
{
    if(argc > 1 && !parse_args(argc, argv)){
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./gst_camera -g  //camera use gst show!" << std::endl;
        std::cerr << "./gst_camera -r  //camera use rtsp streaming!" << std::endl;
        std::cerr << "./gst_camera -gr //camera use gst show and rtsp streaming!" << std::endl;
        std::cerr << "./gst_camera -o  //camera use opencv show!" << std::endl;
    }
    std::cout << "Start GstCamera!" << std::endl;
    signal(SIGINT, signal_handler);
    shm_init();
    pthread_t thread_capture;
    pthread_create(&thread_capture, NULL, captureImage, NULL);
    if(USE_GST_RTSP){
        pthread_t thread_gst_rtsp;
        pthread_create(&thread_gst_rtsp, NULL, gst_rtsp, NULL);
    }
    if(USE_GST_SHOW){
        pthread_t thread_gst_show;
        pthread_create(&thread_gst_show, NULL, gst_show, NULL);
    }

    int key = -1;
    Mat frame;
    Mat resized_image;
    while(!stop_rc){
        if(sem_wait(&(shared->sem)) == -1)
        {
            printf("P ERROR!\n");
            exit(EXIT_FAILURE);
        }
        memcpy(BGR_IMG, shared->Buffer, WIDTH*HEIGHT*3);
        sem_post(&shared->sem);
        frame = Mat(HEIGHT, WIDTH, CV_8UC3, BGR_IMG);
        //http server
        //cv::resize(frame, frame, Size(1280, 720), 0, 0, cv::INTER_LINEAR);
        send_mjpeg(frame, 8090, 400000, 70);
        //usleep(10);

        //save pic
        //cv::imwrite("test.jpg", frame);
        //usleep(10000);
    }
    return 0;
}

