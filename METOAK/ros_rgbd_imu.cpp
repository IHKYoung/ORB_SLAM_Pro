#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;
using namespace std::chrono;

// IMU和相机的时间偏移
float shift = 0;

class ImuGrabber
{
public:
    ImuGrabber(){};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb) : mpSLAM(pSLAM),
                                                                 mpImuGb(pImuGb) {}

    void GrabImageRgb(const sensor_msgs::ImageConstPtr &msg);

    void GrabImageDepth(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgRgbBuf, imgDepthBuf;
    std::mutex mBufMutexRgb, mBufMutexDepth;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD_Inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 从rosrun ORB_SLAM3 RGBD_Inertial之后开始数参数个数
    if (argc < 3 || argc > 4)
    {
        cerr << endl
             << "Usage: rosrun orb_slam3 ros_rgbd_imu path_to_vocabulary path_to_settings (open_viewer_or_not)" << endl;
        ros::shutdown();
        return 1;
    }
    // 默认关闭显示
    bool bUseViewer = false;
    ros::param::get("~useViewer", bUseViewer);
    if (argc == 4)
    {
        bUseViewer = std::string(argv[3]) == std::string("true");
        ROS_INFO("Open Viewer is %s", argv[3]);
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_RGBD, true);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb);

    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe("/mo_cam/imu_info", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_rgb = n.subscribe("/mo_cam/image_color", 100, &ImageGrabber::GrabImageRgb, &igb);
    ros::Subscriber sub_img_depth = n.subscribe("/mo_cam/image_depth", 100, &ImageGrabber::GrabImageDepth, &igb);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    time_t fileTime = system_clock::to_time_t(system_clock::now());
    std::stringstream fileTimeFormat;
    fileTimeFormat << std::put_time(std::localtime(&fileTime), "-%Y-%m-%d-%H-%M-%S");
    string fileLocation;
    ros::param::get("~trajectoryFileLocation", fileLocation);

    string fileTrajectorName = fileLocation + "rosRGBDICameraTrajectory" + fileTimeFormat.str() + ".txt";
    string fileKFTrajectorName = fileLocation + "rosRGBDIKeyFrameTrajectory" + fileTimeFormat.str() + ".txt";

    SLAM.SaveTrajectoryMO(fileTrajectorName);
    SLAM.SaveKeyFrameTrajectoryMO(fileKFTrajectorName);

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImageRgb(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexRgb.lock();
    if (!imgRgbBuf.empty())
        imgRgbBuf.pop();
    imgRgbBuf.push(img_msg);
    mBufMutexRgb.unlock();
}

void ImageGrabber::GrabImageDepth(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexDepth.lock();
    if (!imgDepthBuf.empty())
        imgDepthBuf.pop();
    imgDepthBuf.push(img_msg);
    mBufMutexDepth.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    return cv_ptr->image.clone();
}

void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    while (1)
    {
        cv::Mat imRgb, imDepth;
        double tImRgb = 0, tImDepth = 0;
        if (!imgRgbBuf.empty() && !imgDepthBuf.empty() && !mpImuGb->imuBuf.empty())
        {
            tImRgb = imgRgbBuf.front()->header.stamp.toSec();
            tImDepth = imgDepthBuf.front()->header.stamp.toSec();

            this->mBufMutexDepth.lock();
            while ((tImRgb - tImDepth) > maxTimeDiff && imgDepthBuf.size() > 1)
            {
                imgDepthBuf.pop();
                tImDepth = imgDepthBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexDepth.unlock();

            this->mBufMutexRgb.lock();
            while ((tImDepth - tImRgb) > maxTimeDiff && imgRgbBuf.size() > 1)
            {
                imgRgbBuf.pop();
                tImRgb = imgRgbBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRgb.unlock();

            if ((tImRgb - tImDepth) > maxTimeDiff || (tImDepth - tImRgb) > maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if (tImRgb > mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;

            this->mBufMutexRgb.lock();
            imRgb = GetImage(imgRgbBuf.front());
            imgRgbBuf.pop();
            this->mBufMutexRgb.unlock();

            this->mBufMutexDepth.lock();
            imDepth = GetImage(imgDepthBuf.front());
            imgDepthBuf.pop();
            this->mBufMutexDepth.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tImRgb + shift)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x,
                                    mpImuGb->imuBuf.front()->linear_acceleration.y,
                                    mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x,
                                    mpImuGb->imuBuf.front()->angular_velocity.y,
                                    mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();

            mpSLAM->TrackRGBD(imRgb, imDepth, tImRgb, vImuMeas);

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}