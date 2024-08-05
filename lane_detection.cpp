/*
 * Copyright 2021 iwatake2222
 * Copyright 2024 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * i.MX Lane Detection application using GStreamer + NNStreamer
 *
 * Targets: i.MX8M Plus & i.MX93
 *
 * i.MX lane detection demo demonstrates the machine learning (ML) capabilities
 * of i.MX SoC by using the neural processing unit (NPU) to accelerate two deep 
 * learning vision-based models. Together, these models detect up to four lane 
 * lines and object (eg:Car,truck,person,etc.) on the road.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <glib.h>
#include <gst/gst.h>
#include <cairo.h>
#include <cairo-gobject.h>
#include <opencv2/opencv.hpp>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <vector>
#include <numeric> 
#include <algorithm> 
#include <cstdlib>
#include <sys/unistd.h>
#include <stdexcept>
/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

// /**
//  * @brief Macro to check error case.
//  */
#define _check_cond_err(cond) \
  do { \
    if (!(cond)) { \
      _print_log ("app failed! [line : %d]", __LINE__); \
      goto error; \
    } \
  } while (0)

using namespace cv;
using namespace std;

// Data structure for app
typedef struct
{
  GMainLoop *loop; /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for data stream */
  GstBus *bus; /**< gst bus for data pipeline */
  gboolean running; /**< true when app is running */
} AppData;

// Data for pipeline and result.
static AppData g_app;

/**
 * @brief Free resources in app data.
 */
static void
_free_app_data (void)
{
  if (g_app.loop) {
    g_main_loop_unref (g_app.loop);
    g_app.loop = NULL;
  }

  if (g_app.bus) {
    gst_bus_remove_signal_watch (g_app.bus);
    gst_object_unref (g_app.bus);
    g_app.bus = NULL;
  }

  if (g_app.pipeline) {
    gst_object_unref (g_app.pipeline);
    g_app.pipeline = NULL;
  }

}

/**
 * @brief Function to print error message.
 */
static void
_parse_err_message (GstMessage * message)
{
  gchar *debug;
  GError *error;

  g_return_if_fail (message != NULL);

  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error (message, &error, &debug);
      break;

    case GST_MESSAGE_WARNING:
      gst_message_parse_warning (message, &error, &debug);
      break;

    default:
      return;
  }

  gst_object_default_error (GST_MESSAGE_SRC (message), error, debug);
  g_error_free (error);
  g_free (debug);
}

/**
 * @brief Function to print qos message.
 */
static void
_parse_qos_message (GstMessage * message)
{
  GstFormat format;
  guint64 processed;
  guint64 dropped;

  gst_message_parse_qos_stats (message, &format, &processed, &dropped);
  _print_log ("format[%d] processed[%" G_GUINT64_FORMAT "] dropped[%"
      G_GUINT64_FORMAT "]", format, processed, dropped);
}

/**
 * @brief Callback for message.
 */
static void
_message_cb (GstBus * bus, GstMessage * message, gpointer user_data)
{
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_EOS:
      _print_log ("received eos message");
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_ERROR:
      _print_log ("received error message");
      _parse_err_message (message);
      g_main_loop_quit (g_app.loop);
      break;

    case GST_MESSAGE_WARNING:
      _print_log ("received warning message");
      _parse_err_message (message);
      break;

    case GST_MESSAGE_STREAM_START:
      _print_log ("received start message");
      break;

    case GST_MESSAGE_QOS:
      _parse_qos_message (message);
      break;

    default:
      break;
  }
}

/**
 * @brief Set window title.
 * @param name GstXImageSink element name
 * @param title window title
 */
static void
_set_window_title (const gchar * name, const gchar * title)
{
  GstTagList *tags;
  GstPad *sink_pad;
  GstElement *element;

  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), name);

  g_return_if_fail (element != NULL);

  sink_pad = gst_element_get_static_pad (element, "sink");

  if (sink_pad) {
    tags = gst_tag_list_new (GST_TAG_TITLE, title, NULL);
    gst_pad_send_event (sink_pad, gst_event_new_tag (tags));
    gst_object_unref (sink_pad);
  }

  gst_object_unref (element);
}



// The demo setting init
struct Settings {
  string video_name = "lane_detection.mov";
  string demo_source = "video";
  string car_centor = "320.0";
  string car_length = "280.0";
};
Settings s;

// End flags for the tensorsink element callbacks of the two models
gboolean new_data_updata0 = TRUE ; 
gboolean new_data_updata1 = TRUE ; 

std::vector<int> class_ids;
std::vector<float> score_vec;
std::vector<Rect> objects;
std::vector<int> indices;
float score_threshold = 0.15; 
float nms_threshold = 0.3; 
void * video_caps = NULL;
typedef std::vector<std::pair<int32_t, int32_t>> Line;
std::vector<Line> lane_points_mat;
std::vector<bool> lanes_detected;
float col_sample_w = float(8.070707070707071);
std::vector<int32_t>tusimple_row_anchor = { 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
	116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
	168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
	220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
	272, 276, 280, 284};
static std::vector<cv::Scalar> lane_colors={cv::Scalar(0,0,255),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,255,255)};
vector<Scalar> color;
int kNumGriding=101;
int kNumClassPerLine=56;
int kNumLine=4;
double ctime00 = 0.0;
double ptime00 = 0.0;
double fps00 = 0.0;
struct Pointt {
  double x;
  double y;
};
std::vector<Pointt> left_points;
std::vector<Pointt> right_points;
std::pair<int, int> lane_left_u = std::make_pair(0, 0);
std::pair<int, int> lane_left_d = std::make_pair(0, 0);
std::pair<int, int> lane_right_u = std::make_pair(0, 0);
std::pair<int, int> lane_right_d = std::make_pair(0, 0);
float left_loc=-1.0;
float right_loc=-1.0;
float lane_right_up = 0.0;
float lane_left_up = 0.0;
float car_left_loc = 180.0;
float car_right_loc = 460.0;    

String hostname;
String get_hostname(){
  char host_name[256];
  if (gethostname(host_name, sizeof(host_name)) == 0) {
    // obtained the host name
    std::cout << "Hostname: " << host_name << std::endl;
  }
  String hostname(host_name);
  return hostname;
}
// Calculate the mean of a set of points
double average(const std::vector<Pointt>& points, double (Pointt::*member)) {
    return std::accumulate(points.begin(), points.end(), 0.0,
                            [member](double sum, const Pointt& point) {
                                return sum + point.*member;
                            }) / points.size();
}

// Calculate the slope and intercept of the best fit line using the least squares method
std::pair<double, double> leastSquaresFit(const std::vector<Pointt>& points) {
    double xAvg = average(points, &Pointt::x);
    double yAvg = average(points, &Pointt::y);

    double numerator = 0.0;
    double denominator = 0.0;

    for (const auto& point : points) {
        numerator += (point.x - xAvg) * (point.y - yAvg);
        denominator += (point.x - xAvg) * (point.x - xAvg);
    }

    double slope = numerator / denominator;
    double intercept = yAvg - slope * xAvg;

    return {slope, intercept};
}

static void
_prepare_overlay_cb (GstElement * overlay, GstCaps * caps, gpointer user_data)
{
  video_caps = caps;
}

std::vector<std::string> _className = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush"
};

static std::vector<float> Softmax_0(const std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<float> res(100 * 56 * 4);
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += std::exp(val_list.at(i * (num_j * num_k) + j * num_k + k));
            }
            for (int32_t i = 0; i < num_i; i++) {
                float v =  std::exp(val_list.at(i * (num_j * num_k) + j * num_k + k)) / sum;
                res.at(i * (num_j * num_k) + j * num_k + k) = v;
            }
        }
    }
    return res;
}

static std::vector<float> MulSum(const std::vector<float>& val_list_0, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<float> res(num_j * num_k);
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += val_list_0.at(i * (num_j * num_k) + j * num_k + k) * float((i+1));
            }
            res.at(j * num_k + k) = sum;
        }

    }
    return res;
}

/**
 * @brief Callback for tensor sink signal.
 */
static void
_new_data_cb1 (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  if (g_app.running) {
    GstMemory *mem0;
    GstMapInfo info0;
    GstMemory *mem1;
    GstMapInfo info1;
    GstMemory *mem2;
    GstMapInfo info2;
    GstMemory *mem3;
    GstMapInfo info3;            
    guint num_mems;
    num_mems = gst_buffer_n_memory (buffer);
    if (num_mems!=4){
        return ;
    }

    mem0 = gst_buffer_peek_memory (buffer, 0);
    if (!gst_memory_map (mem0, &info0, GST_MAP_READ)){
        return ;
    }
    mem1 = gst_buffer_peek_memory (buffer, 1);
    if (!gst_memory_map (mem1, &info1, GST_MAP_READ)){
        return ;
    }    

    mem2 = gst_buffer_peek_memory (buffer, 2);
    if (!gst_memory_map (mem2, &info2, GST_MAP_READ)){
        return ;
    }
    mem3 = gst_buffer_peek_memory (buffer, 3);
    if (!gst_memory_map (mem3, &info3, GST_MAP_READ)){
        return ;
    }  

    float* mem_boxes = (gfloat *)info0.data;
    float* mem_detections = (gfloat *)info1.data;
    float* mem_scores = (gfloat *)info2.data;
    float* mem_num = (gfloat *)info3.data;    
    int class_num = int(mem_num[0]);

    int boxes_arraySize = sizeof(mem_boxes) / sizeof(mem_boxes[0]); 
    int detections_arraySize = sizeof(mem_detections) / sizeof(mem_detections[0]); 
    int scores_arraySize = sizeof(mem_scores) / sizeof(mem_scores[0]); 
    int num_arraySize = sizeof(mem_num) / sizeof(mem_num[0]); 

    class_ids.clear();
    score_vec.clear();
    objects.clear();

    for(int i = 0; i < class_num; i++){
        float score = mem_scores[i];
        if (score >= score_threshold) {
            int c = int(mem_detections[i]);
            float ymin = mem_boxes[4 * i + 0];
            float xmin = mem_boxes[4 * i + 1];
            float ymax = mem_boxes[4 * i + 2];
            float xmax = mem_boxes[4 * i + 3];    
            float x = xmin * 640.0;  // model width
            float y = ymin * 480.0;    
            int left = MAX(int(x), 0); 
            int top = MAX(int(y), 0);
            float w = (xmax - xmin) * 640.0;
            float h = (ymax - ymin) * 480.0;       
            class_ids.push_back(c);
            score_vec.push_back(score);
            objects.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));	            
        }
                      
    }

	indices.clear();
	cv::dnn::NMSBoxes(objects, score_vec, score_threshold, nms_threshold, indices); //nms
    gst_memory_unmap (mem0, &info0);
    gst_memory_unmap (mem1, &info1);
    gst_memory_unmap (mem2, &info2);
    gst_memory_unmap (mem3, &info3);
    new_data_updata1 = TRUE ;
  }
}



bool isPairNotEmpty(const std::pair<int, int>& p) {
    return p.first != 0 && p.second != 0;
}


/**
 * @brief Callback for tensor sink signal.
 */
static void
_new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
{
  if (g_app.running) {
    GstMemory *mem0;
    GstMapInfo info0;
    guint num_mems;
    num_mems = gst_buffer_n_memory (buffer);
    if (num_mems!=1){
        return ;
    }

    mem0 = gst_buffer_peek_memory (buffer, 0);
    if (!gst_memory_map (mem0, &info0, GST_MAP_READ)){
        return ;
    } 

    float* output_data = (gfloat *)info0.data;

    lanes_detected.clear();
    lane_points_mat.clear();
    std::vector<float> output_raw_val(output_data, output_data + 101 * 56 * 4 + 1);
    std::vector<float> output_change(kNumGriding * kNumClassPerLine * kNumLine);
    for (int32_t i = 0; i < 101; i++) {
        for (int32_t j = 0; j < 56; j++) {
            for (int32_t k = 0; k < 4; k++) {
                output_change.at(i*56*4+(55-j)*4+k) = output_raw_val.at(i*56*4 + j*4 + k);
            }
        }
    }
    std::vector<float> prob = Softmax_0(output_change, kNumGriding - 1, kNumClassPerLine, kNumLine);
    std::vector<float> loc = MulSum(prob, kNumGriding - 1, kNumClassPerLine, kNumLine);
    for (int32_t j = 0; j < kNumClassPerLine; j++) {
        for (int32_t k = 0; k < kNumLine; k++) {
            float max_val = -999;
            int32_t max_index = 0;
            for (int32_t i = 0; i < kNumGriding; i++) {
                float val = output_change.at(i * (kNumClassPerLine * kNumLine) + j * kNumLine + k);
                if (val > max_val) {
                    max_val = val;
                    max_index = i;
                }
            }
            if (max_index == kNumGriding - 1) {
                loc.at(j * kNumLine + k) = 0.0;
            }
        }
    }
    lane_left_u = std::make_pair(0, 0);
    lane_left_d = std::make_pair(0, 0);
    lane_right_u = std::make_pair(0, 0);
    lane_right_d = std::make_pair(0, 0);
    left_loc = -999.0;
    right_loc = -999.0;
    left_points.clear();
    right_points.clear();
    for (int32_t k = 0; k < kNumLine; k++) {
        Line lane_points;
        float sum = 0;
        for(int32_t j = 0; j < kNumClassPerLine; j++){
            sum += loc.at(j*kNumLine+k);
        }
        if(sum > 2.0){
            lanes_detected.push_back(1);
            for(int32_t j = 0; j < kNumClassPerLine; j++){
                if(loc.at(j*kNumLine+k)>0){
                    int x = static_cast<int32_t>(loc.at(j*kNumLine+k) * col_sample_w * 640.0 / 800) - 1 ;
                    int y = static_cast<int32_t>(480.0 * (tusimple_row_anchor.at(56-1-j)) / 288 ) -1;
                    if(k ==1 && y >= 250.0 && y <= 360.0 ){
                        Pointt left_dot = {double(x),480.0-y};
                        left_points.push_back(left_dot);
                    }

                    if(k ==2 && y >= 250.0 && y <= 360.0 ){
                        Pointt right_dot = {double(x),480.0-y};
                        right_points.push_back(right_dot);
                    }
                    lane_points.push_back({ x, y });                    
                } 
            }
            if(!left_points.empty()){
              auto [left_slope, left_intercept] = leastSquaresFit(left_points);
              left_loc = - (left_intercept / left_slope);
              lane_left_up = (220.0 - left_intercept) /  left_slope;                     
            }

            if(!right_points.empty()){
              auto [right_slope, right_intercept] = leastSquaresFit(right_points);
              right_loc = - (right_intercept / right_slope);
              lane_right_up = (220.0 - right_intercept) /  right_slope;   
            }
       
        }
        else {
            lanes_detected.push_back(0);
        }
        lane_points_mat.push_back(lane_points);
    }
    gst_memory_unmap (mem0, &info0);
    new_data_updata0 = TRUE ;
  }
}

/**
 * @brief Callback to draw the overlay.
 */
static void
_draw_overlay_cb (GstElement * overlay, cairo_t * cr, guint64 timestamp,
    guint64 duration, gpointer user_data)
{
    if (video_caps != NULL   ){
        while(!new_data_updata0 || !new_data_updata1){
        g_usleep (10);
        }
        //predecision detection
        Rect holeImgRect(0, 0, 640, 480);
        for (size_t i = 0; i < indices.size(); ++i){
          int idx = indices[i];
          cv::Rect box = objects[idx] & holeImgRect;
          if(class_ids[idx] == 2 ){   
            if (box.width > 450 ){
                continue;
            }
          }       
          // Remove unsupported categories
          if(class_ids[idx] == 9){
            continue;
          }                  
          if(class_ids[idx] > 12){
            continue;
          }            
          /* draw rectangle */
          cairo_rectangle (cr, box.x, box.y, box.width, box.height);
          cairo_set_source_rgba (cr, float(color[class_ids[idx]][2])/255.0, float(color[class_ids[idx]][1])/255.0, float(color[class_ids[idx]][0])/255.0,1);
          cairo_set_line_width (cr, 5);
          cairo_stroke (cr);
          cairo_fill (cr);
          string label0 = format("%.2f", score_vec[idx]);
          if (!_className.empty() && class_ids[idx] <= 12 )
          {   
              label0 = _className[class_ids[idx]] + ":" + label0;
          }
          const char * label =  label0.c_str();

          /* draw title */
          cairo_set_line_width (cr, 7);
          cairo_set_font_size(cr, 15);
          cairo_move_to (cr, box.x , box.y - 4);
          cairo_show_text(cr,label);
          cairo_stroke (cr);
          cairo_fill (cr);
        }

        //lane detection
        for(int i=0;i<lane_points_mat.size();i++){
            Line out = lane_points_mat.at(i);
            cairo_set_source_rgba (cr, float(lane_colors[i][2])/255.0, float(lane_colors[i][1])/255.0, float(lane_colors[i][0])/255.0,1);
            if(out.size()){                        
                for(int j=0;j<out.size();j++ ){
                    cairo_arc (cr, out[j].first, out[j].second, 3, 0, 2 * M_PI);
                    cairo_fill (cr);
                }
            }
        }

        cairo_set_source_rgba (cr, 1, 1, 0,0.5);
        cairo_set_line_width (cr, 10);
        cairo_move_to (cr, left_loc, 480.0);
        cairo_line_to (cr, right_loc, 480.0);
        cairo_line_to (cr, lane_right_up, 260.0);
        cairo_line_to (cr, lane_left_up, 260.0);
        cairo_close_path(cr);
        cairo_fill (cr);


        //draw the land offset
        String deviation = "Deviation: "; 

        //draw the ADAS box:
        cairo_rectangle (cr, 0, 0, 190, 240);
        cairo_set_source_rgba (cr, 0.0, 0.0, 0.0,0.5);
        cairo_fill (cr);  
        cairo_move_to (cr, 0, 240);
        cairo_line_to (cr, 190, 240);
        cairo_line_to (cr, 190, 0);
        cairo_set_source_rgb (cr, 0.0, 0.0, 1.0);
        cairo_set_line_width (cr, 0.7);
        cairo_stroke (cr);

        // ADAS
        String adas = "LANE DETECTION" ;
        cairo_set_source_rgb (cr, 1, 1, 1);
        cairo_move_to (cr, 8 , 30);
        cairo_set_font_size(cr, 20);
        cairo_show_text(cr, adas.c_str());
        cairo_stroke (cr);
        cairo_fill (cr);  

        cairo_move_to (cr, 0, 50);
        cairo_line_to (cr, 190, 50);
        cairo_set_source_rgb (cr, 0.0, 0.0, 1.0);
        cairo_stroke (cr);

        //draw the fps
        ctime00 = (double)cv::getTickCount();	
        fps00 = 1.0 / ( (ctime00 - ptime00)/ cv::getTickFrequency())  ;
        ptime00 = ctime00;

        String ss = "FPS: " ;
        String hhh = ss +  to_string(int(fps00));
        cairo_set_source_rgb (cr, 1, 1, 1);
        cairo_move_to (cr, 5 , 80);
        cairo_set_font_size(cr, 17);
        cairo_show_text(cr, hhh.c_str());

        cairo_stroke (cr);
        cairo_fill (cr);   

        int obj_number = indices.size();

        //draw the OBJ
        String obj = "OBJ: " ;
        obj = obj +  to_string(obj_number);
        cairo_set_source_rgb (cr, 1, 1, 1);
        cairo_move_to (cr, 5 , 120);
        cairo_set_font_size(cr, 17);
        cairo_show_text(cr, obj.c_str());

        cairo_stroke (cr);
        cairo_fill (cr);  

        //draw the lane warning
        cairo_set_source_rgb (cr, 1, 1, 1);
        cairo_set_font_size(cr, 17);
        cairo_move_to (cr, 5 , 160);
        cairo_show_text(cr, deviation.c_str());
        cairo_stroke (cr);
        cairo_fill (cr);   


        if(right_loc != -999.0 && left_loc != -999.0 ){
          if(right_loc > car_right_loc && left_loc < car_left_loc){
            cairo_set_source_rgb (cr, 0,1,0);
            cairo_set_font_size(cr, 17);
            cairo_move_to (cr, 5 , 200);
            cairo_show_text(cr, "Good Lane Keeping");
            cairo_stroke (cr);
            cairo_fill (cr);            
          }
          else{
            cairo_set_source_rgb (cr, 1,0,0);
            cairo_set_font_size(cr, 17);
            cairo_move_to (cr, 5 , 200);
            cairo_show_text(cr, "Warning! OFF Lane");
            cairo_stroke (cr);
            cairo_fill (cr);           
          }         
        }
        else{
          cairo_set_source_rgb (cr, 0,1,0);
          cairo_set_font_size(cr, 17);
          cairo_move_to (cr, 5 , 200);
          cairo_show_text(cr, "NO Lane Detection");
          cairo_stroke (cr);
          cairo_fill (cr);              
        }
        if(right_loc != -999.0 && left_loc != -999.0 ){
          float deviation_direction = left_loc + (right_loc - left_loc) / 2.0;    
          float deviation_per = (deviation_direction - (car_right_loc - car_left_loc)) / (right_loc - left_loc);   //和车的中心比较
          if(deviation_per >= 0.0){  
            cairo_set_source_rgb (cr, 1,1,1);
            cairo_set_font_size(cr, 17);
            cairo_move_to (cr, 100 , 160);
            cairo_show_text(cr, ("[L]" + to_string(int(deviation_per * 100.0)) + "%").c_str() );
            cairo_stroke (cr);
            cairo_fill (cr);            
          }
          else if(deviation_per < 0.0){
            cairo_set_source_rgb (cr, 1,1,1);
            cairo_set_font_size(cr, 17);
            cairo_move_to (cr, 100 , 160);
            cairo_show_text(cr, ("[R]" + to_string(int(-deviation_per * 100.0)) + "%").c_str() );         
            cairo_stroke (cr);
            cairo_fill (cr);   
          }
        }
    } 

  new_data_updata0 = FALSE; 
  new_data_updata1 = FALSE; 
}

void printHelp() {
    std::cout << "Usage: " << "lane_detection aplication example" << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, Display this help and exit\n";
    std::cout << "  -s, set video source or camera source default by video source (eg: -s camera or -s video)\n";
    std::cout << "  -v, The video name. Must set video name if use your video (eg: -s video -v lane_detection.mp4)\n";
    std::cout << "  -c, The centor pixel of the car default by 320.0 (eg: -c 320.0)\n";
    std::cout << "  -l, The length of the car default by 280.0 (eg: -l 280.0)\n";
}

/**
 * @brief Main function.
 */
int
main (int argc, char **argv)
{
  while (1)
  {  
    static struct option long_options[] = {
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;
    int c;
    c = getopt_long(argc, argv,
                    "c:s:v:l:h", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
	  case 'c':
        s.car_centor = optarg;
        break;
	  case 's':
        s.demo_source = optarg;
        break;	
	  case 'v':
        s.video_name = optarg;
        break;
	  case 'l':
        s.car_length = optarg;
        break;           
	  case 'h':
      printHelp();
      exit(0);
      break;
    case '?':
        exit(-1);
      default:
        exit(-1);
    }
  }

  srand(time(0));
  for (int i = 0; i < 80; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(Scalar(b, g, r));
  }

  gchar *str_pipeline;
  gulong handle_id;
  gulong handle_id_0;
  gulong handle_id_1;
  
  guint timer_id = 0;
  GstElement *element;
  GstMessage *msg;

  _print_log ("start app..");

  /* init app variable */
  g_app.running = FALSE;

  /* init gstreamer */
  gst_init (&argc, &argv);

  /* main loop */
  g_app.loop = g_main_loop_new (NULL, FALSE);
  _check_cond_err (g_app.loop != NULL);       

  car_left_loc = std::stof(s.car_centor)  -  std::stof(s.car_length) / 2;
  car_right_loc = std::stof(s.car_centor)  +  std::stof(s.car_length) / 2;
  hostname = get_hostname();

  if (!hostname.compare("imx93evk")){
    if (!s.demo_source.compare("camera")){
      str_pipeline = 
        g_strdup_printf
        ("v4l2src name=cam_src device=/dev/video0 ! imxvideoconvert_pxp ! video/x-raw,width=640, height=480, framerate=15/1,format=BGRx ! "
          "tee name=t0 t0. ! imxvideoconvert_pxp ! video/x-raw, width=800, height=288 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
          " tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,mul:0.01735207,add:-2.017699 ! tensor_filter framework=tensorflow-lite model=\"./models/model_integer_quant_vela.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libethosu_delegate.so "
          " silent=FALSE name=tensor_filter0 latency=1 ! tensor_sink name=tensor_sink0 t0. ! " 
          "tee name=t1  t1. ! imxvideoconvert_pxp ! video/x-raw, width=300, height=300 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
          " tensor_converter ! tensor_filter framework=tensorflow-lite model=\"./models/mobilenet_ssd_v2_coco_quant_postprocess_vela.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libethosu_delegate.so "
          " silent=FALSE name=tensor_filter1 latency=1 ! tensor_sink name=tensor_sink1 t1. ! "
          "imxvideoconvert_pxp ! cairooverlay name=tensor_res ! queue max-size-buffers=2 leaky=2 ! waylandsink "
          );    
    }
    else{
      str_pipeline =
        g_strdup_printf
          ("filesrc location=%s ! qtdemux ! avdec_h264 ! imxvideoconvert_pxp ! video/x-raw,width=640, height=480 ! "
          "tee name=t0 t0. ! imxvideoconvert_pxp ! video/x-raw, width=800, height=288 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
          " tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,mul:0.01735207,add:-2.017699 ! tensor_filter framework=tensorflow-lite model=\"./models/model_integer_quant_vela.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libethosu_delegate.so "
          " silent=FALSE name=tensor_filter0 latency=1 ! tensor_sink name=tensor_sink0 t0. ! " 
          "tee name=t1  t1. ! imxvideoconvert_pxp ! video/x-raw, width=300, height=300 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !  "
          " tensor_converter ! tensor_filter framework=tensorflow-lite model=\"./models/mobilenet_ssd_v2_coco_quant_postprocess_vela.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libethosu_delegate.so "
          " silent=FALSE name=tensor_filter1 latency=1 ! tensor_sink name=tensor_sink1 t1. ! "
          "imxvideoconvert_pxp ! cairooverlay name=tensor_res ! queue max-size-buffers=2 leaky=2 ! waylandsink ", s.video_name.c_str()

          );        
    }    
  }  
  else if (!hostname.compare("imx8mpevk")){
    if (!s.demo_source.compare("camera")){
      str_pipeline = 
        g_strdup_printf
        ("v4l2src name=cam_src device=/dev/video3 ! imxvideoconvert_g2d ! video/x-raw,width=640, height=480, framerate=30/1,format=BGRx ! "
          "tee name=t0 t0. ! imxvideoconvert_g2d ! video/x-raw, width=800, height=288 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
          " tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,mul:0.01735207,add:-2.017699 ! tensor_filter framework=tensorflow-lite model=\"./models/model_integer_quant.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libvx_delegate.so "
          " silent=FALSE name=tensor_filter0 latency=1 ! tensor_sink name=tensor_sink0 t0. ! " 
          "tee name=t1  t1. ! imxvideoconvert_g2d ! video/x-raw, width=300, height=300 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
          " tensor_converter ! tensor_filter framework=tensorflow-lite model=\"./models/mobilenet_ssd_v2_coco_quant_postprocess.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libvx_delegate.so "
          " silent=FALSE name=tensor_filter1 latency=1 ! tensor_sink name=tensor_sink1 t1. ! "
          "imxvideoconvert_g2d ! cairooverlay name=tensor_res ! queue max-size-buffers=2 leaky=2 ! waylandsink "
          );    
    }
    else{
      str_pipeline =
        g_strdup_printf
          ("filesrc location=%s ! qtdemux ! h264parse ! vpudec ! imxvideoconvert_g2d ! video/x-raw,width=640, height=480 ! "
          "tee name=t0 t0. ! imxvideoconvert_g2d ! video/x-raw, width=800, height=288 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !"
           " tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,mul:0.01735207,add:-2.017699 ! tensor_filter framework=tensorflow-lite model=\"./models/model_integer_quant.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libvx_delegate.so "
          " silent=FALSE name=tensor_filter0 latency=1 ! tensor_sink name=tensor_sink0 t0. ! " 
          "tee name=t1  t1. ! imxvideoconvert_g2d ! video/x-raw, width=300, height=300 ! queue max-size-buffers=2 leaky=2 ! videoconvert ! video/x-raw,format=RGB !  "
          " tensor_converter ! tensor_filter framework=tensorflow-lite model=\"./models/mobilenet_ssd_v2_coco_quant_postprocess.tflite\" accelerator=true:npu custom=Delegate:External,ExtDelegateLib:libvx_delegate.so "
          " silent=FALSE name=tensor_filter1 latency=1 ! tensor_sink name=tensor_sink1 t1. ! "
          "imxvideoconvert_g2d ! cairooverlay name=tensor_res ! queue max-size-buffers=2 leaky=2 ! waylandsink", s.video_name.c_str()
          );        
    }  
  }
  else{
    throw std::runtime_error("Error: Please use the i.mx8mp or i.mx93 running the demo");
  }


  _print_log ("%s\n", str_pipeline);

  g_app.pipeline = gst_parse_launch (str_pipeline, NULL);
  g_free (str_pipeline);
  _check_cond_err (g_app.pipeline != NULL);

  /* bus and message callback */
  g_app.bus = gst_element_get_bus (g_app.pipeline);
  _check_cond_err (g_app.bus != NULL);

  gst_bus_add_signal_watch (g_app.bus);
  handle_id = g_signal_connect (g_app.bus, "message",
      (GCallback) _message_cb, NULL);
  _check_cond_err (handle_id > 0);

  /* tensor sink signal : new data callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink0");
  handle_id = g_signal_connect (element, "new-data",
      (GCallback) _new_data_cb, NULL);
  gst_object_unref (element);
  _check_cond_err (handle_id > 0);

  /* tensor sink signal : new data callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_sink1");
  handle_id = g_signal_connect (element, "new-data",
      (GCallback) _new_data_cb1, NULL);
  gst_object_unref (element);
  _check_cond_err (handle_id > 0);

 /* tensor_res(cairooverlay) signal: callback */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "tensor_res");
  handle_id_0 = g_signal_connect (element, "draw",
      (GCallback) _draw_overlay_cb, NULL);
  handle_id_1 = g_signal_connect (element, "caps-changed",
      (GCallback) _prepare_overlay_cb, NULL);      
  gst_object_unref (element);
  _check_cond_err (handle_id_0 > 0);
  _check_cond_err (handle_id_1 > 0);  

  /* start pipeline */
  gst_element_set_state (g_app.pipeline, GST_STATE_PLAYING);

  g_app.running = TRUE;

  /* set window title */
  _set_window_title ("img_tensor", "NNStreamer Example");

  /* run main loop */
  g_main_loop_run (g_app.loop);

  /* quit when received eos or error message */
  g_app.running = FALSE;

  /* cam source element */
  element = gst_bin_get_by_name (GST_BIN (g_app.pipeline), "cam_src");

  gst_element_set_state (element, GST_STATE_READY);
  gst_element_set_state (g_app.pipeline, GST_STATE_READY);

  g_usleep (200 * 1000);

  gst_element_set_state (element, GST_STATE_NULL);
  gst_element_set_state (g_app.pipeline, GST_STATE_NULL);

  g_usleep (200 * 1000);
  gst_object_unref (element);

  msg =
      gst_bus_timed_pop_filtered (g_app.bus, GST_CLOCK_TIME_NONE,
      (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  /* See next tutorial for proper error message handling/parsing */
  if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
    g_error ("An error occurred! Re-run with the GST_DEBUG=*:WARN environment "
        "variable set for more details.");
  }

  /* Free resources */
  gst_message_unref (msg);



error:
  _print_log ("close app..");

  if (timer_id > 0) {
    g_source_remove (timer_id);
  }

  _free_app_data ();
  return 0;
}