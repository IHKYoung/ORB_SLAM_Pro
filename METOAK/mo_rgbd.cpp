//
// Created by young on 22-12-1.
//
#include "common_function.h"
#include "opencv2/opencv.hpp"

#include "System.h"
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// 处理ctrl+c中断
#include <csignal>
// add pose
#include <geometry_msgs/PoseStamped.h>

using namespace cv;
using namespace Eigen;
using namespace std;
using namespace std::chrono;


int32_t main(int32_t argc, char **argv)
{
    ros::init(argc, argv, "mo_rgbd");

    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 3)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 mo_rgbd path_to_vocabulary path_to_settings)"
             << endl;
        return 1;
    }

    std::string camPath;
    int fps_;

    nh.param<std::string>("cam_path", camPath, "/dev/video0");
    nh.param<int>("fps", fps_, 30);

    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/vslam/pose", 1, true);

    // Init ORB-SLAM3
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);

    printf("The version of SDK is %s\n", moGetSdkVersion());
    MO_CAMERA_HANDLE hCameraHandle = MO_INVALID_HANDLE;

    int32_t n32Result = 0;
    // Open camera by path
    n32Result = moOpenUVCCameraByPath(camPath.c_str(), &hCameraHandle);
    if (0 != n32Result)
    {
        printf("Error, moOpenUVCCameraByPath Failed %d\n", n32Result);
        return -1;
    }

    // Get video resolution
    uint16_t u16VideoFrameWidth = 0;
    uint16_t u16VideoFrameHeight = 0;
    n32Result = moGetVideoResolution(hCameraHandle, &u16VideoFrameWidth, &u16VideoFrameHeight);
    if (0 != n32Result)
    {
        printf("Error, moGetVideoResolution Failed %d\n", n32Result);
        return -2;
    }

    // Get the BxF(baseline * focus) and baseline of camera
    float pfBxf, pfBase; // pfBxf = Focus * pfBase
    n32Result = moGetBxfAndBase(hCameraHandle, &pfBxf, &pfBase);
    if (0 != n32Result)
    {
        printf("Error, moGetBxfAndBase Failed %d\n", n32Result);
        return -3;
    }

    // 设置补光灯
    moSetFilllightType(hCameraHandle, MFT_OFF);

    uint64_t u64ImageFrameNum = 0;
    uint8_t *pu8FrameBuffer = nullptr;
    // for RGBD
    uint16_t *pu16RGBDDisparityData = nullptr;
    uint8_t *pu8RGBDYUVI420Img = nullptr;

    // Set video mode to RGBD DENSE mode
    n32Result = moSetVideoMode(hCameraHandle, MVM_RGBD);
    if (0 != n32Result)
    {
        printf("Error: moSetVideoMode Failed %d\n", n32Result);
        return -4;
    }

    ros::Rate loop_rate(fps_);
    while (ros::ok())
    {
        // Get current video frame
        n32Result = moGetCurrentFrame(hCameraHandle, &u64ImageFrameNum, &pu8FrameBuffer);
        if (0 != n32Result)
        {
            printf("Error, moGetCurrentFrame Failed.\n");
            continue;
        }

        ros::Time msg_time = ros::Time().now();
        uint64_t nowTimestamp = msg_time.toNSec() / 1e6;
        n32Result = moGetRGBDImage(hCameraHandle, pu8FrameBuffer, &pu16RGBDDisparityData,
                                   &pu8RGBDYUVI420Img);

        // cout << "Timestamp: "<< nowTimestamp << endl;

        Mat matYUV2BGR; // 12bits / 8bits
        Mat matYUVI420(u16VideoFrameHeight * 1.5, u16VideoFrameWidth, CV_8UC1, pu8RGBDYUVI420Img);
        cvtColor(matYUVI420, matYUV2BGR, COLOR_YUV2BGR_I420);

        Mat matDisparity(u16VideoFrameHeight, u16VideoFrameWidth, CV_16UC1, pu16RGBDDisparityData);

        Mat matShowDisparity(u16VideoFrameHeight, u16VideoFrameWidth, CV_8UC3);
        GetDisparityImage(matShowDisparity.rows, matShowDisparity.cols, matShowDisparity.step, pu16RGBDDisparityData,
                          matShowDisparity.data);
        // imshow("Disparity Show", matShowDisparity);
        Mat matDepth = cv::Mat::zeros(u16VideoFrameHeight, u16VideoFrameWidth, CV_16UC1);
        for (int row = 0; row < matDepth.rows; row++)
        {
            for (int col = 0; col < matDepth.cols; col++)
            {
                u_short disp = matDisparity.ptr<ushort>(row)[col];
                if (disp == 0)
                    continue;
                matDepth.ptr<u_short>(row)[col] = 32.0 * pfBxf / disp;
            }
        }
        // imshow("Depth Show", matDepth);
        int trackingState = -1;
        Sophus::SE3f Tcw = SLAM.TrackRGBDStatus(matYUV2BGR, matDepth, nowTimestamp, trackingState);
        Sophus::SE3f Twc_old = Tcw.inverse();
        // publish pose
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.frame_id = "camera_link";
        pose_msg.header.stamp = msg_time;

        // 调整坐标系, 保持右手系, x-front, y-left, z-up
        Matrix3f R_old = Twc_old.rotationMatrix();
        Vector3f t_old = Twc_old.translation();
        // cout << "R old = \n" << R_old << endl;

        // Y 顺时针 90
        Matrix3f rotationMatrixY;
        rotationMatrixY = AngleAxisf(M_PI_2, Vector3f::UnitY());
        // 新X 逆时针 90
        Matrix3f rotationMatrixX;
        rotationMatrixX = AngleAxisf(-1.0 * M_PI_2, Vector3f::UnitX());

        Matrix3f R_new = rotationMatrixX * rotationMatrixY * R_old;
        Vector3f t_new = rotationMatrixX * rotationMatrixY * t_old;
        // cout << "R new = \n" << R_new << endl;

        Sophus::SE3f Twc(R_new, t_new);
        pose_msg.pose.position.x = Twc.translation().x();
        pose_msg.pose.position.y = Twc.translation().y();
        pose_msg.pose.position.z = Twc.translation().z();

        pose_msg.pose.orientation.w = Twc.unit_quaternion().coeffs().w();
        pose_msg.pose.orientation.x = Twc.unit_quaternion().coeffs().x();
        pose_msg.pose.orientation.y = Twc.unit_quaternion().coeffs().y();
        pose_msg.pose.orientation.z = Twc.unit_quaternion().coeffs().z();
        pose_pub.publish(pose_msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    // Stop all threads
    SLAM.Shutdown();
    // Save camera trajectory
    time_t fileTime = system_clock::to_time_t(system_clock::now());
    std::stringstream fileTimeFormat;
    fileTimeFormat << std::put_time(std::localtime(&fileTime), "-%Y-%m-%d-%H-%M-%S");
    string fileLocation;
    ros::param::get("~trajectoryFileLocation", fileLocation);

    string fileTrajectorName = fileLocation + "moRGBDCameraTrajectory" + fileTimeFormat.str() + ".txt";
    string fileKFTrajectorName = fileLocation + "moRGBDKeyFrameTrajectory" + fileTimeFormat.str() + ".txt";

    SLAM.SaveTrajectoryMO(fileTrajectorName);
    SLAM.SaveKeyFrameTrajectoryMO(fileKFTrajectorName);

    // Close camera
    moCloseCamera(&hCameraHandle);
    return n32Result;
}
