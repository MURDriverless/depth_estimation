// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

// solvePnP is listed in this header file! 
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core.hpp>

#include "ConeROI.hpp"
#include "ConeEst.hpp"

#include "dataStructures.h"
#include "matching2D.hpp"

// add custom keys if you want to use the parser to your liking 
const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{left_image il        |<none>| input image   }"
"{right_image ir       |<none>| input image  }"
"{left_vid v       |<none>| input video   }"
"{device d       |<cpu>| input device   }"
"{is_left l     |<none>| input int      }"
"{right_vid rv   |<none>| input video }"
;
using namespace cv;
using namespace dnn;
using namespace std;
using namespace cv::xfeatures2d;

/**
 * \brief Compute and draw the epipolar lines in two images
 *      associated to each other by a fundamental matrix
 *
 * \param title     Title of the window to display
 * \param F         Fundamental matrix
 * \param img1      First image
 * \param img2      Second image
 * \param points1   Set of points in the first image
 * \param points2   Set of points in the second image matching to the first set
 * \param inlierDistance      Points with a high distance to the epipolar lines are
 *                not displayed. If it is negative, all points are displayed
 **/


// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
vector<string> classes_right; 

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs);

void print_num_cones(vector<int> classIds, string image);

void postprocess_yolo_right_img(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs);

void draw_bounding_boxes(vector<int> indices, vector<Rect> boxes, vector<int> classIds, vector<float> confidences, Mat frame);

void compute_bounding_box(const vector<Mat>& outs, Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, vector<int>& centerX, vector<int>& centerY);

void keypoint_detection(vector<int>& indices, vector<Rect>& boxes, Mat& frame, vector<cv::KeyPoint>& final_keypoints, vector<int>& centerX, vector<int>& centerY);

// void average_keypoint(vector<KeyPoint>& input, vector <Keypoint>& output);

// template <typename T1, typename T2>
// static void drawEpipolarLines(const std::string& title, const cv::Matx<T1,3,3> F,
//                 const cv::Mat& img1, const cv::Mat& img2,
//                 const std::vector<cv::Point_<T2>> points1,
//                 const std::vector<cv::Point_<T2>> points2,
//                 const float inlierDistance = -1);

// template <typename T>
// static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float zEst, int show_coordinates); //  float xEst, float yEst);

void print_inference_time(Net net, Mat& frame, string outputFile);

void show_frame(Mat& frame, string frame_name);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{   

    const std::string& calibrationFile = "left_calibration.xml";
    cv::FileStorage fs;
    cv::Size calibSize;
    cv::Mat cameraMatrix; 
    cv::Mat distCoeffs; 

    fs.open(calibrationFile, cv::FileStorage::READ);
    fs["cameraMatrix"]   >> cameraMatrix;
    fs["distCoeffs"]     >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    float imgCenter_x = cameraMatrix.at<double>(0, 2);
    float imgCenter_y = cameraMatrix.at<double>(1, 2);

    float focal_px_x = cameraMatrix.at<double>(0, 0);
    float focal_px_y = cameraMatrix.at<double>(1, 1);

    fs.release();

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    string classesFile = "cones.names"; // use the list of class names for our cones i.e., blue, orange, and yellow 
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = "cpu";
    device = parser.get<String>("device");

    String modelConfiguration = "yolov4-tiny.cfg";
    String modelWeights = "yolov4-tiny-best.weights";

    int left_cam_only = 0;

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    Net net_right = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device == "cpu")
    {
        cout << "Using CPU device" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    int FPS = 10;
    
    string str_right, outputFile_right;
    VideoCapture cap_right;
    VideoWriter video_right;
    Mat frame_right, blob_right;

    int is_left = 0;

    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("left_image"))
        {
            // Open the image file
            str = parser.get<String>("left_image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            // cap.open(str);

            frame = imread(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }

         if (parser.has("right_image"))
        {
            // Open the image file
            str_right = parser.get<String>("right_image");
            ifstream ifile(str_right);
            if (!ifile) throw("error");
            // cap_right.open(str_right);
            frame_right = imread(str_right);

            str_right.replace(str_right.end()-4, str_right.end(), "_yolo_out_cpp.jpg");
            outputFile_right = str_right;
        }
        
        
        if (parser.has("left_vid"))
        {
            // Open the video file (left camera)
            str = parser.get<String>("left_vid");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;

           
            
        }
        
        if (parser.has("right_vid")) {
             // right camera
            if (left_cam_only == 0) {
                str_right = parser.get<String>("right_vid");
                ifstream ifile_right(str_right);
                if (!ifile_right) throw("error");
                cap_right.open(str_right);
                str_right.replace(str_right.end()-4, str_right.end(), "_yolo_out_cpp.avi");
                outputFile_right = str_right;
            }
        }
        // Open the webcam
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    if (parser.has("left_vid")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), FPS, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
        
        if (left_cam_only == 0) {
            video_right.open(outputFile_right, VideoWriter::fourcc('M','J','P','G'), FPS, Size(cap_right.get(CAP_PROP_FRAME_WIDTH), cap_right.get(CAP_PROP_FRAME_HEIGHT)));
        }
    }
    
    // process the images and videos 
    if (parser.has("left_vid")) {

         // Create a window
        static const string kWinName = "Deep learning object detection in OpenCV";
        namedWindow(kWinName, WINDOW_NORMAL);

        // Process frames.
        while (waitKey(1) < 0)
        {   
            // get frame from the video
            cap >> frame;
            cap_right >> frame_right;

            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!" << endl;
                cout << "Output file is stored as " << outputFile << endl;
                waitKey(3000);
                break;
            }

            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            vector<Mat> outs_right;

            // Create a 4D blob from a frame.
            blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net.setInput(blob); // anything to do with "net" means it's dealing with darknet 
            net.forward(outs, getOutputsNames(net));
            

            
            blobFromImage(frame_right, blob_right, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net_right.setInput(blob_right);
            net_right.forward(outs_right, getOutputsNames(net_right));

            // postprocess(frame, outs, frame_right, outs_right, is_left, cameraMatrix, distCoeffs);
            postprocess_yolo_right_img(frame, outs, frame_right, outs_right, is_left, cameraMatrix, distCoeffs);
            
            print_inference_time(net, frame, outputFile);
            print_inference_time(net_right, frame_right, outputFile_right);
            
            // Write the frame with the detection boxes
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);
            if (parser.has("left_image")) imwrite(outputFile, detectedFrame);
            else video.write(detectedFrame);

            Mat detectedFrame_right;
            frame_right.convertTo(detectedFrame_right, CV_8U);
            // imshow("right image", detectedFrame_right);
            // waitKey(0);
            video_right.write(detectedFrame_right);

            imshow(kWinName, detectedFrame);
            
        }
    }
    // supplied images! 
    else {  

        cv::String left_path("../../../../FPS_10_stereo_left/*.png"); 
        cv::String right_path("../../../../FPS_10_stereo_right/*.png"); 
        vector<cv::String> fn;
        vector<cv::String> fn_right;
        // vector<cv::Mat> data;
        cv::glob(left_path,fn,true); // recurse
        cv::glob(right_path, fn_right, true);

        for (size_t k=0; k<fn.size(); ++k)
        {
            cv::Mat im = cv::imread(fn[k]);
            cv::Mat im_right = cv::imread(fn_right[k]);
            
            if (im.empty()) continue; //only proceed if sucsessful
            // you probably want to do some preprocessing

            frame = im; 
            outputFile = fn[k]; 

            frame_right = im_right;
            outputFile_right = fn_right[k];

            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            vector<Mat> outs_right;
            // Create a 4D blob from a frame.
            blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net.setInput(blob);       
            net.forward(outs, getOutputsNames(net));

            blobFromImage(frame_right, blob_right, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net_right.setInput(blob_right);
            net_right.forward(outs_right, getOutputsNames(net_right));
            
            // postprocess(frame, outs, frame_right, outs_right, is_left, cameraMatrix, distCoeffs);
            postprocess_yolo_right_img(frame, outs, frame_right, outs_right, is_left, cameraMatrix, distCoeffs);

            print_inference_time(net, frame, outputFile);
            print_inference_time(net_right, frame_right, outputFile_right);
            
            // show_frame(frame, "left_image");
            // show_frame(frame_right, "right_image");
        }
    }
    
    cap.release();
    cap_right.release();

    if (parser.has("left_vid")) {
        video.release();
        video_right.release();
    } 

    return 0;
}

void show_frame(Mat& frame, string frame_name) {
    // Write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    imshow(frame_name, detectedFrame);
    waitKey(0);
}

void print_inference_time(Net net, Mat& frame, string outputFile) {
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    cout << "Frame: " << outputFile << ", Inference Time: " << t << endl;
}
        
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs)
{   

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    int centerX;
    int centerY;
    int width;
    int height;

    // cout << outs.size() << endl;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {   
                // manually resized the bounding box to encapsulate the entire cone
                // tried training the object detection algorithm but didn't see an improvement in performance. 
                centerX = (int)(data[0] * frame.cols);
                centerY = (int)(data[1] * frame.rows);
                width = (int)(data[2] * frame.cols)/2 - 20;
                height = (int)(data[3] * frame.rows) + 95;
                int left = centerX - width / 2 - 10;
                int top = centerY - height / 2 - 15;
                
                // cout << "width is: " << width << endl;
                // cout << "height is: " << height << endl;

                // cout << "centerX is: " << centerX << endl;
                // cout << "centerY is: " << centerY << endl;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));

                
            }
        }
    }

    
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // added by Kelvin 
    double baseline = 550; // 550mm 
    double f2 = 1499.093; // focal length of right camera = 5mm (in the x axis)
    double f1 = 2724.847; // focal length of left camera in the x axis

    double lImgCenter_x = 548.607;
    double lImgCenter_y = 845.6168;

    double rImgCenter_x = 921.35846;
    double rImgCenter_y = 616.6407;

    // cout << "after rImgCenter" << endl;

    std::vector<std::vector<cv::Point3f>> conePointsVec; 
    
    // define 7 points that shape a cone 
    std::vector<cv::Point3f> conePoints;
    conePoints.push_back(cv::Point3f(0, 0, 0));
    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 300.0f/3.0f * i;

        conePoints.push_back(cv::Point3f( x, y, 0));
        conePoints.push_back(cv::Point3f(-x, y, 0));
    }
    std::vector<cv::Point3f> conePointsBig;
    conePointsBig.push_back(cv::Point3f(0, 0, 0));
    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 505.0f/3.0f * i;

        conePointsBig.push_back(cv::Point3f( x, y, 0));
        conePointsBig.push_back(cv::Point3f(-x, y, 0));
    }

    conePointsVec.push_back(conePoints);
    conePointsVec.push_back(conePointsBig);

    cv::Mat imgGray;
    cv::Mat roi_left_out;

    // crop the left frame with the bounding box of a cone from the left frame 
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        // cout << "idx is: " << idx << endl;

        // try {
        //     cv::Mat roi(frame, box);

        //     if (roi.empty()) {
        //         continue;
        //     }

        //     cv::cvtColor(roi, imgGray, cv::COLOR_BGR2GRAY);

        //     roi_left_out = roi; 

        // } catch (const std::exception& e) { // reference to the base of a polymorphic object
        //     std::cout << "roi not possible" << endl; // information from length_error printed
        //     continue;
        // }
    
        // // perform SIFT on it to find the keypoints 
        // string detectorType = "SIFT";
        // bool visDetector = false; // visualise the results 
    
        // vector<cv::KeyPoint> keypoints;

        // DetectorTypeIndex detectorTypeIndex = getDetectorTypeIndex(detectorType);

        // detKeypointsModern(keypoints, imgGray, detectorTypeIndex, visDetector);

        // vector<cv::Point2f> keypoints_new;

        // if (keypoints.size() == 0) {
        //     continue;
        // }
        // // manually moves the keypoints to the cone 
        // for (size_t i=0; i < keypoints.size(); i++) {

        //     cv::Point2f keypoint = keypoints[i].pt;
        //     keypoint.x = keypoint.x + centerX - 40;
        //     keypoint.y = keypoint.y + centerY - 40; 

        //     // if (keypoint.x > centerX + width/3 || keypoint.x < centerX - width/3) {
        //     //     continue;
        //     // }

        //     // if (keypoint.y > centerY + height/3 || keypoint.y < centerY - height/3) {
        //     //     continue;
        //     // }

        //     keypoints_new.push_back(keypoint);
        // }

        // // // cout << "size of keypoints new is: " << keypoints.size() << endl;

        // vector<cv::KeyPoint> keypoints_temp;
        // cv::KeyPoint::convert(keypoints_new, keypoints_temp);

        // // // cout << keypoints_new << endl;

        // // ****************************************************************************************************************
        // // draw the keypoints on the entire frame
        // // drawKeypoints(frame, keypoints_temp, frame, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // Mat tvec;
        // Mat rvec;

        // conePoints = conePointsVec[0]; 

        // const std::vector<cv::Point3f> &conePts = conePoints;

        // std::vector<cv::Point2f> keypoints_points;

        // // cv::KeyPoint::convert(keypoints_temp, keypoints_points);
        // cv::KeyPoint::convert(keypoints, keypoints_points);

        // // cout << keypoints_points << endl;

        // if (keypoints_points.size() < 7) {
        //     cout << "less than 7" << endl;
        //     continue;
        // }

        // std::vector<cv::Point2f> keypoints_points_tmp;

        // // cout << keypoints_points << endl;

        // keypoints_points_tmp.push_back(keypoints_points[0]);
        // keypoints_points_tmp.push_back(keypoints_points[1]);
        // keypoints_points_tmp.push_back(keypoints_points[2]);
        // keypoints_points_tmp.push_back(keypoints_points[3]);
        // keypoints_points_tmp.push_back(keypoints_points[4]);
        // keypoints_points_tmp.push_back(keypoints_points[5]);
        // keypoints_points_tmp.push_back(keypoints_points[6]);

        // // cout << conePts << endl;

        // // this won't work unless conePts and keypoints_points_tmp both have 7 elements each 
        // bool ret = cv::solvePnP(conePts, keypoints_points_tmp, cameraMatrix, distCoeffs, rvec, tvec, false, SolvePnPMethod::SOLVEPNP_IPPE);

        // // estimate depth 
        // double est_depth = 0;

        // // this line won't work till the solvePnP line works. sigh. 
        // if (tvec.size().empty()) {
        //     cout << "tvec size is [0x0]" << endl;
        //     continue;
        // }
        
        // // cout << tvec.size() << endl;

        // est_depth = tvec.at<double>(2, 0);

        // if (est_depth < 0) {
        //     est_depth = abs(est_depth);
        // }
    
        // cout << "est_depth is: " << est_depth << endl;

        // cv::Mat imgGray_right;

        // cv::Mat roi_right_out;
        // cv::Rect projRect_out;

        // //******************************************************************************************
        // // bounding box for right image 
        // try {

        //      cv::Rect projRect(box);

        //     int x_p = box.x - lImgCenter_x;
        //     int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - baseline);

        //     projRect.x = x_pp + rImgCenter_x - 200;

        //     // cout << "after projRect.x" << endl;
            
        //     cv::Mat roi_right(frame_right, projRect);

        //     if (roi_right.empty()) {
        //         continue;
        //     }

        //     cv::cvtColor(roi_right, imgGray_right, cv::COLOR_BGR2GRAY);

        //     roi_right_out = roi_right;
        //     projRect_out = projRect;

        // } catch (const std::exception& e) { // reference to the base of a polymorphic object
        //     std::cout << "roi not possible" << endl; // information from length_error printed
        //     continue;
        // }
        
        // vector<cv::KeyPoint> keypoints_right;

        // detKeypointsModern(keypoints_right, imgGray_right, detectorTypeIndex, visDetector);

        // // cout << "after keypoint_right" << endl;

        // vector<cv::Point2f> keypoints_new_right;

        // if (keypoints_right.size() < 7) {
        //     cout << "less than 7" << endl;
        //     continue;
        // }

        // // cout << "size of keypoints: " << keypoints.size() << endl;

        // // manually moves the keypoints to the cone 
        // for (size_t i=0; i < keypoints_right.size(); i++) {
        //     cv::Point2f keypoint = keypoints_right[i].pt;
        //     keypoint.x = keypoint.x + centerX - 40;
        //     keypoint.y = keypoint.y + centerY - 40; 

        //     // if (keypoint.x > centerX + width/3 || keypoint.x < centerX - width/3) {
        //     //     continue;
        //     // }

        //     // if (keypoint.y > centerY + height/3 || keypoint.y < centerY - height/3) {
        //     //     continue;
        //     // }

        //     keypoints_new_right.push_back(keypoint);
        // }

        // // cout << "after resizing keypoint" << endl;

        // // cout << "size of keypoints new is: " << keypoints.size() << endl;

        // vector<cv::KeyPoint> keypoints_temp_right;
        // cv::KeyPoint::convert(keypoints_new_right, keypoints_temp_right);

        // // draw the keypoints on the entire frame
        // // drawKeypoints(frame_right, keypoints_temp_right, frame_right, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // std::vector<cv::KeyPoint> featureKeypoints1;
        // std::vector<cv::KeyPoint> featureKeypoints2;
        // cv::Mat descriptors1;
        // cv::Mat descriptors2;

        // std::vector<cv::DMatch> matches;
        // std::vector<cv::DMatch> matchesFilt;

        // // cv::Ptr<cv::Feature2D> featureDetector;
        // cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
        
        // cv::Ptr<cv::FastFeatureDetector> FeatureDetector = cv::FastFeatureDetector::create(40);

        // FeatureDetector->detect(roi_left_out, featureKeypoints1);
        // FeatureDetector->detect(roi_right_out, featureKeypoints2);
        
        // cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(32);

        // brief->compute(roi_left_out, featureKeypoints1, descriptors1);
        // brief->compute(roi_right_out, featureKeypoints2, descriptors2);

        // drawKeypoints(frame, featureKeypoints1, frame, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // // imshow("Image", frame);
        // // waitKey(1000);

        // // ********************************************************************************************************************
        // // /*
        // // No descriptor in left or right frame, either due to insufficient light.
        // // Or plain ground textures, expecially in synthetic data
        // // */
        // if (descriptors1.empty() || descriptors2.empty()) {
        //     cout << "descriptors empty" << endl;
        //     continue;
        // }

        // cout << "before match ##################################################" << endl;

        // // cout << descriptors1 << endl;
        // // cout << descriptors2 << endl;

        // if (descriptors2.size() != descriptors1.size()) {
        //     cout << "description matrix size doesn't match" << endl;
        //     continue;
        // }

        // descriptorMatcher->match(descriptors1, descriptors2, matches);

        // cout << "after match ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl; 

        // // // this code is causing seg faults atm... 
        // // // Filters for horizontal-ish matches only 
        // uint32_t yDelta = projRect_out.height * 0.1;
        // for (const cv::DMatch &match : matches) {
        //     if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
        //         matchesFilt.push_back(match);
        //     }
        // }

        // // // Check if no valid matches
        // // if (matchesFilt.size() == 0) {
        // //     cout << "matchesFilt size 0 " << endl;
        // //     continue;
        // // }
        
        // // cout << "matches" << endl;

        // // std::vector<float> disparity;
        // // for (const cv::DMatch &match : matchesFilt) {
        // //     float x1 = featureKeypoints1[match.queryIdx].pt.x;
        // //     x1 += box.x;
        // //     x1 -= lImgCenter_x;

        // //     float x2 = featureKeypoints2[match.trainIdx].pt.x;
        // //     x2 += projRect.x;
        // //     x2 -= rImgCenter_x;

        // //     disparity.push_back(x1*f2/f1 - x2);
        // // }

        // // // // Performance loss from not using a sorted heap should be negligable
        // // std::sort(disparity.begin(), disparity.end());

        // // float medDisp = disparity[(int) disparity.size()/2];
        // // float zEst = baseline*f2/medDisp;

        // // cout << "zEst is: " << zEst << endl;

        // // added by Kelvin 
        // // disparity = (left_box.x - lImgCenter_x)*f2/f1 - (right_box.x - rImgCenter_x);
        // // float zEst = _baseline*f2/disparity;
        // float zEst = 4400; // 4400mm
        // float xEst = 0;
        // float yEst = 0;

        // // if (is_left == 1) {
        //     // left image 
        //     xEst = zEst*(box.x + box.width/2 - lImgCenter_x)/f1; // Andrew's code has a minus sign in front of zEst
        //     yEst = zEst*(box.y + box.height - lImgCenter_y)/f1; // Andrew's code has a minus sign in front of zEst
        // // }
        // // else {
        //     // right image 
        //     xEst = zEst*(box.x + box.width/2 - rImgCenter_x)/f2; // Andrew's code has a minus sign in front of zEst
        //     yEst = zEst*(box.y + box.height - rImgCenter_y)/f2; // Andrew's code has a minus sign in front of zEst
        // // }
        // // convert to m
        // zEst = zEst/1000; 
        // xEst = xEst/1000;
        // yEst = yEst/1000;

        float zEst = 0;

        int show_coordinates = 0;

        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, zEst, show_coordinates); //, xEst, yEst);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float zEst, int show_coordinates) // , float xEst, float yEst)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    // string conePoseEst = format("(%.2f, %.2f, %.2f)m", xEst, yEst, zEst);
    string conePoseEst = format("(%.2f)m", zEst);

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    Size conePoseEstSize = getTextSize(conePoseEst, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    // print cone pose estimation 
    // rectangle(frame, Point(left, bottom + round(2*conePoseEstSize.height)), Point(left + round(1.5*conePoseEstSize.width), bottom + baseLine), Scalar(255, 255, 255), FILLED);
    rectangle(frame, Point(left, bottom), Point(left, bottom + baseLine), Scalar(255, 255, 255), FILLED);

    if (show_coordinates == 1) {
        putText(frame, conePoseEst, Point(left, bottom + round(2*conePoseEstSize.height)), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
    }
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }

    // for (int i = 0; names.size(); i++) {
    //     cout << names[i] << endl;
    // }
    
    return names;
}



// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess_yolo_right_img(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs) {   

    // added by Kelvin 
    double baseline = 550; // 550mm 
    double f2 = 1499.093; // focal length of right camera = 5mm (in the x axis)
    double f1 = 2724.847; // focal length of left camera in the x axis

    double lImgCenter_x = 548.607;
    double lImgCenter_y = 845.6168;

    double rImgCenter_x = 921.35846;
    double rImgCenter_y = 616.6407;

    // ****************************************** LEFT IMAGE *****************************************************
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    vector<int> centerX;
    vector<int> centerY;

    compute_bounding_box(outs, frame, classIds, confidences, boxes, centerX, centerY);
    print_num_cones(classIds, "Left Image");

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    vector<cv::KeyPoint> final_keypoints;
    keypoint_detection(indices, boxes, frame, final_keypoints, centerX, centerY);

    //******************************************* RIGHT IMAGE *****************************************************
    cv::Mat imgGray_right;
    cv::Mat roi_right_out;
    cv::Rect projRect_out;

    vector<int> centerX_right;
    vector<int> centerY_right;

    vector<int> classIds_right;
    vector<float> confidences_right;
    vector<Rect> boxes_right;

    compute_bounding_box(outs_right, frame_right, classIds_right, confidences_right, boxes_right, centerX_right, centerY_right);
    print_num_cones(classIds_right, "Right Image");

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    vector<int> indices_right;
    NMSBoxes(boxes_right, confidences_right, confThreshold, nmsThreshold, indices_right);

    vector<cv::KeyPoint> final_keypoints_right;
    keypoint_detection(indices_right, boxes_right, frame_right, final_keypoints_right, centerX_right, centerY_right);

    // cv::KeyPoint final_left_keypoint;
    // cv::KeyPoint final_right_keypoint;

    // average_keypoint(final_keypoints, )
    // std::vector<float> disparity;

    // float temp;
    // float temp_min = 1000000;

    // for (size_t i=0; i < keypoints_left.size(); i++) {
    //     cv::Point2f keypoint_left = keypoints_left[i];

    //     int idx = indices[i];
    //     Rect box = boxes[idx];

    //     for (size_t j=0; j < keypoints_right.size(); j++) {
    //         cv::Point2f keypoint_right = keypoints_right[j];

    //         int idx_right = indices_right[j];
    //         Rect box_right = boxes_right[idx_right];

    //         temp = (box.x + keypoint_left.x - lImgCenter_x)*f2/f1 - (box_right.x + keypoint_right.x - rImgCenter_x); 

    //         if (temp < temp_min) {
    //             temp_min = temp; 
    //         }
    //     }

    //     disparity.push_back(temp_min);
    //     temp_min = 100000000;
    // }

    // float disparity_avg = (abs(disparity[0]) + abs(disparity[1]))/2;

    draw_bounding_boxes(indices, boxes, classIds, confidences, frame);
    draw_bounding_boxes(indices_right, boxes_right, classIds_right, confidences_right, frame_right);
  
}

void print_num_cones(vector<int> classIds, string image) {

    cout << image << endl;

    int num_blue = 0;
    int num_yellow = 0;

    for (int i = 0; i < classIds.size(); i++) {

        // blue cone
        if (classIds[i] == 0) {
            num_blue = num_blue + 1;
        }
        else if (classIds[i] == 2) {
            num_yellow = num_yellow + 1;
        }
    }
    
    cout << "Total Blue Cones: " << num_blue << endl;
    cout << "Total Yellow Cones: " << num_yellow << endl;
}

// void average_keypoint(vector<KeyPoint>& input, vector <KeyPoint>& output) {
    
//     cv::Point2f avg_keypoint;
//     cv::KeyPoint avg_keypoint_KeyPoint;
//     avg_keypoint.x = 0;
//     avg_keypoint.y = 0;

//     for (size_t i=0; i < input.size(); i++) {
//         cv::Point2f keypoint = input[i].pt;
//         avg_keypoint.x = avg_keypoint.x + keypoint.x; 
//         avg_keypoint.y = avg_keypoint.y + keypoint.y; 
//     }

//     avg_keypoint.x = avg_keypoint.x/input.size();
//     avg_keypoint.y = avg_keypoint.y/input.size();

//     cv::KeyPoint::convert(avg_keypoint, avg_keypoint_KeyPoint);

//     output.push_back(avg_keypoint_KeyPoint;
// }

void draw_bounding_boxes(vector<int> indices, vector<Rect> boxes, vector<int> classIds, vector<float> confidences, Mat frame) {
    for (size_t i = 0; i < indices.size(); ++i) {   
    // cout << "disparity calculations" << endl;

    int idx = indices[i];
    Rect box = boxes[idx];
    
    // cout << disparity_avg << endl;

    // float zEst = baseline*f2/disparity[idx];
    // float zEst = baseline*f2/disparity_avg;
    // float xEst = 0;
    // float yEst = 0;

    // xEst = zEst*(box.x + box.width/2 - lImgCenter_x)/f1; // Andrew's code has a minus sign in front of zEst
    // yEst = zEst*(box.y + box.height - lImgCenter_y)/f1; // Andrew's code has a minus sign in front of zEst

    // // convert to m
    // zEst = zEst/1000; 
    // xEst = xEst/1000;
    // yEst = yEst/1000;

    // if (xEst < 0) {
    //     xEst = abs(xEst);
    // }

    // if (yEst < 0) {
    //     yEst = abs(yEst);
    // }

    // zEst = zEst + 1;
    int zEst = 0;

    int show_coordinates = 0; 

    // loop again to draw the bounding boxes around the cones 
    drawPred(classIds[idx], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, frame, zEst, show_coordinates); // , xEst, yEst);
    // cout << "drew pred" << endl;
    }
}

void compute_bounding_box(const vector<Mat>& outs, Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, vector<int>& centerX, vector<int>& centerY) {

    int width;
    int height;
    int centerY_temp;
    int centerX_temp;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {   
                // manually resized the bounding box to encapsulate the entire cone
                // tried training the object detection algorithm but didn't see an improvement in performance. 
                centerX_temp = (int)(data[0] * frame.cols);
                centerY_temp = (int)(data[1] * frame.rows);

                width = (int)(data[2] * frame.cols)/2;
                height = (int)(data[3] * frame.rows) + 95;
                int left = centerX_temp - width / 2;
                int top = centerY_temp - height / 2 - 15;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
                centerX.push_back(centerX_temp);
                centerY.push_back(centerY_temp);
                
            }
        }
    }
} 

void keypoint_detection(vector<int>& indices, vector<Rect>& boxes, Mat& frame, vector<cv::KeyPoint>& final_keypoints, vector<int>& centerX, vector<int>& centerY) {
    
    // the keypoints for the left and right cone are stacked in one vector... 
    cv::Mat imgGray;
    cv::Mat roi_left_out;
    vector<cv::Point2f> keypoints_Point2f;

    string detectorType = "SIFT";
    bool visDetector = false; // visualise the results 
    vector<cv::KeyPoint> keypoints;

    // crop the left frame with the bounding box of a cone from the left frame 
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];

        try {
            cv::Mat roi(frame, box);

            if (roi.empty()) {
                continue;
            }

            cv::cvtColor(roi, imgGray, cv::COLOR_BGR2GRAY);
            roi_left_out = roi; 

        } catch (const std::exception& e) { // reference to the base of a polymorphic object
            std::cout << "roi not possible" << endl; // information from length_error printed
            continue;
        }

        // perform SIFT on it to find the keypoints 
        DetectorTypeIndex detectorTypeIndex = getDetectorTypeIndex(detectorType);
        detKeypointsModern(keypoints, imgGray, detectorTypeIndex, visDetector);

        if (keypoints.size() == 0) {
            continue;
        }

            // manually moves the keypoints to the cone 
        for (size_t i=0; i < keypoints.size(); i++) {
            cv::Point2f single_keypoint = keypoints[i].pt;
            single_keypoint.x = single_keypoint.x + centerX[idx] - 60;
            single_keypoint.y = single_keypoint.y + centerY[idx] - 85; 
            keypoints_Point2f.push_back(single_keypoint);
        }
        
        cv::KeyPoint::convert(keypoints_Point2f, final_keypoints);
        drawKeypoints(frame, final_keypoints, frame, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }

}    


// used to draw epipolar lines but doesn't work 
// template <typename T1, typename T2>
// static void drawEpipolarLines(const std::string& title, const cv::Matx<T1,3,3> F,
//                 const cv::Mat& img1, const cv::Mat& img2,
//                 const std::vector<cv::Point_<T2>> points1,
//                 const std::vector<cv::Point_<T2>> points2,
//                 const float inlierDistance = -1)
// {
//   CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
//   cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
//   cv::Rect rect1(0,0, img1.cols, img1.rows);
//   cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
//   /*
//    * Allow color drawing
//    */
//   if (img1.type() == CV_8U)
//   {
//     cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
//     cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
//   }
//   else
//   {
//     img1.copyTo(outImg(rect1));
//     img2.copyTo(outImg(rect2));
//   }
//   std::vector<cv::Vec<T2,3>> epilines1, epilines2;
//   cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
//   cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
//   CV_Assert(points1.size() == points2.size() &&
//         points2.size() == epilines1.size() &&
//         epilines1.size() == epilines2.size());
 
//   cv::RNG rng(0);
//   for(size_t i=0; i<points1.size(); i++)
//   {
//     if(inlierDistance > 0)
//     {
//       if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
//         distancePointLine(points2[i], epilines1[i]) > inlierDistance)
//       {
//         //The point match is no inlier
//         continue;
//       }
//     }
//     /*
//      * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
//      */
//     cv::Scalar color(rng(256),rng(256),rng(256));
 
//     cv::line(outImg(rect2),
//       cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
//       cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
//       color);
//     cv::circle(outImg(rect1), points1[i], 3, color, -1, LINE_AA);
 
//     cv::line(outImg(rect1),
//       cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
//       cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
//       color);
//     cv::circle(outImg(rect2), points2[i], 3, color, -1, LINE_AA);
//   }
//   cv::imshow(title, outImg);
//   cv::waitKey(1);
// }

// template <typename T>
// static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line)
// {
//   //Line is given as a*x + b*y + c = 0
//   return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
//       / std::sqrt(line(0)*line(0)+line(1)*line(1));
// }


//    avg_keypoint.x = 0;
//    avg_keypoint.y = 0;

  // manually moves the keypoints to the cone 
    // for (size_t i=0; i < keypoints.size(); i++) {
    //     cv::Point2f keypoint = keypoints[i].pt;
    //     avg_keypoint.x = avg_keypoint.x + keypoint.x; 
    //     avg_keypoint.y = avg_keypoint.y + keypoint.y; 
    // }

    // avg_keypoint.x = avg_keypoint.x/keypoints.size();
    // avg_keypoint.y = avg_keypoint.y/keypoints.size();

    // keypoints_left.push_back(avg_keypoint);


    // // the 0 states to load the images as grayscale 
    // cv::Mat img1 = cv::imread("1m_CameraLeft0036_L.png", 0);
    // cv::Mat img2 = cv::imread("1m_CameraRight0036_R.png", 0);

    // //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    // int minHessian = 400;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints1, keypoints2;
    // Mat descriptors1, descriptors2;
    // detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    // detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
    // //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector< std::vector<DMatch> > knn_matches;
    // matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.5f;
    // std::vector<DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }
    // //-- Draw matches
    // Mat img_matches;
    // // drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // //-- Show detected matches
    // // imshow("Good Matches", img_matches );
    // // waitKey();
    // // return 0;

    // vector<Point2f> points1; 
    // KeyPoint::convert(keypoints1, points1);

    // vector<Point2f> points2; 
    // KeyPoint::convert(keypoints2, points2);

    // long num_matches = good_matches.size();
    // vector<Point2f> matched_points1;
    // vector<Point2f> matched_points2;

    // for (int i=0;i<num_matches;i++)
    // {
    //     int idx1=good_matches[i].trainIdx; // left image
    //     int idx2=good_matches[i].queryIdx; // right image 
    //     matched_points1.push_back(points1[idx1]);
    //     matched_points2.push_back(points2[idx2]);
    // }

    // vector<char> inliers(matched_points1.size());

    // // cv::Mat F = findFundamentalMat(matched_points1, matched_points2, cv::FM_RANSAC, 3, 0.99, inliers);
    // cv::Mat fundamentalMatrix= cv::findFundamentalMat(matched_points1, matched_points2, cv::FM_8POINT);

    // cv::Mat leftLines, rightLines;

    // cv::computeCorrespondEpilines(matched_points1, 1, fundamentalMatrix, leftLines);
    // cv::computeCorrespondEpilines(matched_points2, 2, fundamentalMatrix, rightLines);

    // cv::Mat left_image11, right_image11, left_image22, right_image22;
    // // drawlines(img1, img2, leftLines, matched_points1, matched_points2, left_image11, right_image11);
    // // drawlines(img2, img1, rightLines, matched_points2, matched_points1, left_image22, right_image22);
    
    // // result
    // // imshow("left_image 11", left_image11);
    // // imshow("right_image 22", right_image22);
	

    // // drawMatches(img1, matched_points1, img2, matched_points2, good_matches, img_matches, Scalar::all(-1),
    // //            Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // // imshow("Good Matches", img_matches );
    // // waitKey();
    // // return 0;

    // // Mat img_matches;
    // // drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), inliers, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // // imshow("Matched points", img_matches);
    // // waitKey();
    // return 0;

    //************************************************************************************************************

// void featureDetectorExperimentation()
// {
//     // source: https://stackoverflow.com/questions/39887357/featuredetectors-not-in-opencv-3-0-0 
//     cv::Mat image = imread("bird.jpg");

//     std::vector<KeyPoint> keypoints;
//     cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(40);
//     // Above line compiler error: "Error    1   error C2259: 'cv::FastFeatureDetector' : cannot instantiate abstract class"

//     fast->detect(image, keypoints);

//     drawKeypoints(image, keypoints, image, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

//     imshow("Image", image);
//     waitKey(1000);
// }

// void drawlines(Mat img1, Mat img2, Mat lines, vector<Point2f> pts1, vector<Point2f> pts2, Mat left_image11, Mat right_image11) {
    
//     cv::Size s = img1.size();
//     int rows = s.height;
//     int cols = s.width;

//     int c = cols;  

//     Mat img1_out, img2_out;

//     cv::cvtColor(img1, img1_out, cv::COLOR_GRAY2RGB);
//     cv::cvtColor(img2, img2_out, cv::COLOR_GRAY2RGB);

//     for (int i = 0; i < lines.total(); i++) {
//         std::cout << "hi" << std::endl;
//     }


//     // for i=1:numel(lines)
//     //     clr = randi([0 255], [1 3], 'uint8');
//     //     r = lines{i};
//     //     p1 = uint32([0, -r(3)/r(2)]);
//     //     p2 = uint32([c, -(r(3)+r(1)*c)/r(2)]);
//     //     img1 = cv.line(img1, p1, p2, 'Color',clr, 'LineType','AA');
//     //     img1 = cv.circle(img1, pts1{i}, 5, 'Color',clr, 'Thickness',-1);
//     //     img2 = cv.circle(img2, pts2{i}, 5, 'Color',clr, 'Thickness',-1);
//     // end
//     return;
// }