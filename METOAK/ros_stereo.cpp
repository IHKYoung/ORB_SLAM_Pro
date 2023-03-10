#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "System.h" // ORB_SLAM3

// add pose
#include <opencv2/core/eigen.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Geometry>

using namespace std;
using namespace std::chrono;
using namespace Eigen;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ros::Publisher pose_pub) : mpSLAM(pSLAM), mPosePub(pose_pub) {}

    void GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight);

    ORB_SLAM3::System *mpSLAM;
    ros::Publisher mPosePub;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_stereo");
    ros::start();

    if (argc < 3 || argc > 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 ros_stereo path_to_vocabulary path_to_settings (open_viewer_or_not)" << endl;
        ros::shutdown();
        return 1;
    }
    // 默认关闭显示
    bool bUseViewer = false;
    ros::param::get("~useViewer", bUseViewer);
    // 这么写有点问题
    if (argc == 4)
    {
        bUseViewer = std::string(argv[3]) == std::string("true");
        ROS_INFO("Open Viewer is %s", argv[3]);
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, bUseViewer);

    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/mo_cam/image_left", 100);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/mo_cam/image_right", 100);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/vslam/pose", 10, true);

    ImageGrabber igb(&SLAM, pose_pub);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub, right_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    time_t fileTime = system_clock::to_time_t(system_clock::now());
    std::stringstream fileTimeFormat;
    fileTimeFormat << std::put_time(std::localtime(&fileTime), "-%Y-%m-%d-%H-%M-%S");
    string fileLocation;
    ros::param::get("~trajectoryFileLocation", fileLocation);

    string fileTrajectorName = fileLocation + "rosRGBDCameraTrajectory" + fileTimeFormat.str() + ".txt";
    string fileKFTrajectorName = fileLocation + "rosRGBDKeyFrameTrajectory" + fileTimeFormat.str() + ".txt";

    SLAM.SaveTrajectoryMO(fileTrajectorName);
    SLAM.SaveKeyFrameTrajectoryMO(fileKFTrajectorName);

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft, const sensor_msgs::ImageConstPtr &msgRight)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    int trackingState = -1;
    ofstream log_file;
    string logFileLocation;
    ros::param::get("~logFileLocation", logFileLocation);
    log_file.open(logFileLocation + "orb_elapsed_time.txt", ios::app);
    auto orb_start = high_resolution_clock::now();

    Sophus::SE3f Tcw = mpSLAM->TrackStereoStatus(cv_ptrLeft->image, cv_ptrRight->image, cv_ptrLeft->header.stamp.toSec(), trackingState);
    Sophus::SE3f Twc_old = Tcw.inverse();
    // publish pose
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.frame_id = "camera_link";
    pose_msg.header.stamp = cv_ptrLeft->header.stamp;

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
    pose_msg.pose.position.z = trackingState;

    pose_msg.pose.orientation.w = Twc.unit_quaternion().coeffs().w();
    pose_msg.pose.orientation.x = Twc.unit_quaternion().coeffs().x();
    pose_msg.pose.orientation.y = Twc.unit_quaternion().coeffs().y();
    pose_msg.pose.orientation.z = Twc.unit_quaternion().coeffs().z();
    mPosePub.publish(pose_msg);

    auto orb_end = high_resolution_clock::now();
    auto orb_elapsed = duration_cast<milliseconds>(orb_end - orb_start);
    log_file << cv_ptrLeft->header.stamp << " Elapsed Time: " << orb_elapsed.count() << " milliseconds" << endl;
    log_file.close();
}
