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
    int fps = 30;
    ros::init(argc, argv, "mo_stereo");

    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 mo_stereo path_to_vocabulary path_to_settings path_of_video(default /dev/video0)"
             << endl;
        return 1;
    }

    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/vslam/pose", 1, true);

    // Init ORB-SLAM3
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, true);

    printf("The version of SDK is %s\n", moGetSdkVersion());
    MO_CAMERA_HANDLE hCameraHandle = MO_INVALID_HANDLE;

    int32_t n32Result = 0;
    // Open camera by path
    n32Result = moOpenUVCCameraByPath(argv[3], &hCameraHandle);
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

    // 设置补光灯
    moSetFilllightType(hCameraHandle, MFT_OFF);

    uint64_t u64ImageFrameNum = 0;
    uint8_t *pu8FrameBuffer = NULL;
    // for Stereo
    uint8_t *pu8OnlyY = NULL;
    uint8_t *pu8RectifiedRightYUVI420Img = NULL;

    // Set video mode to MVM_RECTIFIED mode
    n32Result = moSetVideoMode(hCameraHandle, MVM_RECTIFIED);
    if (0 != n32Result)
    {
        printf("Error: moSetVideoMode Failed %d\n", n32Result);
        return -4;
    }

    ros::Rate loop_rate(fps);
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
        n32Result = moGetRectifiedImage(hCameraHandle, pu8FrameBuffer, &pu8OnlyY,
                                        &pu8RectifiedRightYUVI420Img);
        // cout << "Timestamp: "<< nowTimestamp << endl;

        Mat matLeftBGR;
        Mat matLeftRectify(u16VideoFrameHeight, u16VideoFrameWidth, CV_8UC1, pu8OnlyY);
        cvtColor(matLeftRectify, matLeftBGR, COLOR_GRAY2BGR);
        Mat matRightBGR; // 1.5 = 12bits / 8bits
        Mat matRightRectify(u16VideoFrameHeight * 1.5, u16VideoFrameWidth, CV_8UC1, pu8RectifiedRightYUVI420Img);
        cvtColor(matRightRectify, matRightBGR, COLOR_YUV2BGR_I420);

        int trackingState = -1;
        Sophus::SE3f Tcw = SLAM.TrackStereoStatus(matLeftBGR, matRightBGR, nowTimestamp, trackingState);
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
        ;

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

    string fileTrajectorName = fileLocation + "moStereoCameraTrajectory" + fileTimeFormat.str() + ".txt";
    string fileKFTrajectorName = fileLocation + "moStereoKeyFrameTrajectory" + fileTimeFormat.str() + ".txt";

    SLAM.SaveTrajectoryMO(fileTrajectorName);
    SLAM.SaveKeyFrameTrajectoryMO(fileKFTrajectorName);

    // Close camera
    moCloseCamera(&hCameraHandle);
    return n32Result;
}
