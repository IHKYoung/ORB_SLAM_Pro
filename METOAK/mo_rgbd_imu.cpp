#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Imu.h>
#include <thread>

#include "mo_stereo_camera_driver_c.h"
#include "mo_camera_driver/moCameraInfo.h"

void ms_sleep(uint16_t u16Millisecond)
{
    struct timeval stTV;
    stTV.tv_sec = 0;
    stTV.tv_usec = u16Millisecond * 1000;
    select(0, NULL, NULL, NULL, &stTV);
}

unsigned char *getColorTable()
{
    int idx;
    // 24576 = 3 * 8192 * sizeof(unsigned char)
    static unsigned char baColorTable[24576];

    baColorTable[0] = 0;
    baColorTable[1] = 0;
    baColorTable[2] = 0;

    for (idx = 1; idx <= 24; idx++)
    {
        baColorTable[idx * 3] = 255;
        baColorTable[idx * 3 + 1] = 255;
        baColorTable[idx * 3 + 2] = 255;
    }

    for (idx = 25; idx <= 40; idx++)
    {
        // 7.9375 = ((255.0 - 128.0) / (40.0 - 24.0))
        baColorTable[idx * 3] = (int)(255 - (int)(7.9375 * (idx - 24)));
        baColorTable[idx * 3 + 1] = (int)(255 - (int)(7.9375 * (idx - 24)));
        baColorTable[idx * 3 + 2] = (int)(255 - (int)(7.9375 * (idx - 24)));
    }

    for (idx = 41; idx <= 64; idx++)
    {
        baColorTable[idx * 3] = (int)(128 + (int)(5.291668 * (idx - 41)));
        baColorTable[idx * 3 + 1] = (int)(128 - (int)(5.291668 * (idx - 41)));
        baColorTable[idx * 3 + 2] = (int)(128 + (int)(5.291668 * (idx - 41)));
    }

    for (idx = 65; idx <= 120; idx++)
    {
        baColorTable[idx * 3] = (int)(255 - (int)(4.553571 * (idx - 64)));
        baColorTable[idx * 3 + 1] = 0;
        baColorTable[idx * 3 + 2] = 255;
    }

    for (idx = 121; idx <= 176; idx++)
    {
        baColorTable[idx * 3] = 0;
        baColorTable[idx * 3 + 1] = (int)(0 + (int)(4.553571 * (idx - 120)));
        baColorTable[idx * 3 + 2] = 255;
    }

    for (idx = 177; idx <= 320; idx++)
    {
        baColorTable[idx * 3] = 0;
        baColorTable[idx * 3 + 1] = 255;
        baColorTable[idx * 3 + 2] = (int)(255 - (int)(1.770833 * (idx - 176)));
    }

    for (idx = 321; idx <= 800; idx++)
    {
        baColorTable[idx * 3] = (int)(0 + (int)(0.53125 * (idx - 320)));
        baColorTable[idx * 3 + 1] = 255;
        baColorTable[idx * 3 + 2] = 0;
    }

    for (idx = 801; (idx <= 2048); idx++)
    {
        baColorTable[idx * 3] = 255;
        baColorTable[idx * 3 + 1] = (int)(255 - (int)(0.204327 * (idx - 800)));
        baColorTable[idx * 3 + 2] = 0;
    }

    for (idx = 2049; idx < 8192; idx++)
    {
        baColorTable[idx * 3] = 255;
        baColorTable[idx * 3 + 1] = 0;
        baColorTable[idx * 3 + 2] = 0;
    }

    return baColorTable;
}

// make the array of colors table
unsigned char *g_caColorTable = getColorTable();

/** \brief Get the image of disparity data
 *
 * \param u16Height - the height of video frame
 * \param u16Width - the widht of video frame
 * \param u16Step - the step length of depth image data buffer
 * \param pu16DepthData - points depth data of video frame
 * \param pu8DepthImage - points colored image data of depth data
 * \return 0 - success, else - failure
 *        -1 - parameter is invalid
 */
int32_t GetDisparityImage(IN const uint16_t u16Height, IN const uint16_t u16Width, IN const uint16_t u16Step,
                          IN const uint16_t *pu16DepthData, OUT uint8_t *const pu8DepthImage)
{
    int32_t n32Result = 0;

    do
    {
        if ((0 >= u16Width) || (0 >= u16Height) || (0 >= u16Step) || (NULL == pu16DepthData) ||
            (NULL == pu8DepthImage))
        {
            n32Result = -1;
            break;
        }

        for (uint16_t u16IdxH = 0; u16IdxH < u16Height; u16IdxH++)
        {
            unsigned char *pu8ImageData = pu8DepthImage + u16IdxH * u16Step;

            for (uint16_t u16IdxW = 0; u16IdxW < u16Width; u16IdxW++)
            {
                uint16_t u32RawDisparityValue = (*(pu16DepthData++)) & ((unsigned short)0x1fff);

                if (0 < u32RawDisparityValue)
                {
                    *(pu8ImageData++) = g_caColorTable[u32RawDisparityValue * 3 + 2];
                    *(pu8ImageData++) = g_caColorTable[u32RawDisparityValue * 3 + 1];
                    *(pu8ImageData++) = g_caColorTable[u32RawDisparityValue * 3];
                }
                else
                {
                    *(pu8ImageData++) = 0;
                    *(pu8ImageData++) = 0;
                    *(pu8ImageData++) = 0;
                }
            }
        }

    } while (0);

    return n32Result;
}

void *publishIMGProcess(void *phCameraHandle, image_transport::Publisher img_pub, image_transport::Publisher depth_pub)
{
    int32_t n32Result = 0;
    uint64_t u64ImageFrameNum = 0;
    uint8_t *pu8FrameBuffer = nullptr;
    // for RGBD
    uint16_t *pu16RGBDDisparityData = nullptr;
    uint8_t *pu8RGBDYUVI420Img = nullptr;

    // Get video resolution
    uint16_t u16VideoFrameWidth = 0;
    uint16_t u16VideoFrameHeight = 0;
    n32Result = moGetVideoResolution(*((MO_CAMERA_HANDLE *)phCameraHandle), &u16VideoFrameWidth, &u16VideoFrameHeight);
    if (0 != n32Result)
    {
        printf("Error, moGetVideoResolution Failed %d\n", n32Result);
        return NULL;
    }
    n32Result = moSetVideoMode(*((MO_CAMERA_HANDLE *)phCameraHandle), MVM_RGBD);
    if (0 != n32Result)
    {
        printf("Error: moSetVideoMode to RGBD(DENSE) failed %d\n", n32Result);
        return NULL;
    }

    float pfBxf, pfBase; // pfBxf = Focus * pfBase
    n32Result = moGetBxfAndBase(*((MO_CAMERA_HANDLE *)phCameraHandle), &pfBxf, &pfBase);
    if (0 != n32Result)
    {
        printf("Error, moGetBxfAndBase Failed %d\n", n32Result);
        return NULL;
    }
    printf("\nBxF = %f\nBaseline = %f\n", pfBxf, pfBase);

    while (ros::ok())
    {
        ms_sleep(10);

        ros::Time time_now = ros::Time::now();
        // Get current video frame
        n32Result = moGetCurrentFrame(*((MO_CAMERA_HANDLE *)phCameraHandle), &u64ImageFrameNum, &pu8FrameBuffer);
        if (0 != n32Result)
        {
            printf("Error: moGetCurrentFrame return %d\n", n32Result);
            continue;
        }

        printf("\nImage>>>\nFrameNum: %lu\n", u64ImageFrameNum);

        n32Result = moGetRGBDImage(*((MO_CAMERA_HANDLE *)phCameraHandle), pu8FrameBuffer, &pu16RGBDDisparityData,
                                   &pu8RGBDYUVI420Img);

        cv::Mat matYUV2BGR; // 12bits / 8bits
        cv::Mat matYUVI420(u16VideoFrameHeight * 1.5, u16VideoFrameWidth, CV_8UC1, pu8RGBDYUVI420Img);
        cvtColor(matYUVI420, matYUV2BGR, cv::COLOR_YUV2BGR_I420);

        cv::Mat matDisparity(u16VideoFrameHeight, u16VideoFrameWidth, CV_16UC1, pu16RGBDDisparityData);

        cv::Mat matShowDisparity(u16VideoFrameHeight, u16VideoFrameWidth, CV_8UC3);
        GetDisparityImage(matShowDisparity.rows, matShowDisparity.cols, matShowDisparity.step, pu16RGBDDisparityData,
                          matShowDisparity.data);

        cv::Mat matDepth = cv::Mat::zeros(u16VideoFrameHeight, u16VideoFrameWidth, CV_16UC1);
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

        std_msgs::Header header;
        header.stamp = time_now;
        header.frame_id = "camera_link";

        cv_bridge::CvImage cv_image = cv_bridge::CvImage(header, "bgr8", matYUV2BGR);
        cv_bridge::CvImage cv_depth = cv_bridge::CvImage(header, "mono16", matDepth);

        img_pub.publish(cv_image.toImageMsg());
        depth_pub.publish(cv_depth.toImageMsg());
        ros::spinOnce();
    }
    return NULL;
}

void *publishIMUProcess(void *phCameraHandle, ros::Publisher imu_pub)
{
    int32_t n32Result = 0;
    mo_imu_data *pstIMUData = NULL;

    while (ros::ok())
    {
        ms_sleep(3);

        ros::Time time_now = ros::Time::now();
        // get current IMU data
        n32Result = moGetIMUData(*((MO_CAMERA_HANDLE *)phCameraHandle), &pstIMUData);
        if (0 != n32Result)
        {
            printf("Error: moGetIMUData return %d\n", n32Result);
            continue;
        }

        sensor_msgs::Imu imu_info;
        imu_info.header.stamp = time_now;
        imu_info.header.frame_id = "camera_link";
        imu_info.linear_acceleration.x = pstIMUData->dAccelX;
        imu_info.linear_acceleration.y = pstIMUData->dAccelY;
        imu_info.linear_acceleration.z = pstIMUData->dAccelZ;

        imu_info.angular_velocity.x = pstIMUData->dGyroX;
        imu_info.angular_velocity.y = pstIMUData->dGyroY;
        imu_info.angular_velocity.z = pstIMUData->dGyroZ;
        imu_pub.publish(imu_info);

        ros::spinOnce();
    }

    return NULL;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mo_rgbd_imu_driver");
    ros::NodeHandle nh;

    std::string cam_path;
    int fps_;

    nh.param<std::string>("cam_path", cam_path, "/dev/video0");
    nh.param<int>("fps", fps_, 30);

    int32_t n32Result = 0;
    MO_CAMERA_HANDLE hCameraHandle = MO_INVALID_HANDLE;

    n32Result = moOpenUVCCameraByPath(cam_path.c_str(), &hCameraHandle);
    if (0 != n32Result)
    {
        printf("Error: moOpenUVCCameraByPath return %d\n", n32Result);
        return -1;
    }

    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise("/mo_cam/image_color", 10);
    image_transport::Publisher depth_pub = it.advertise("/mo_cam/image_depth", 10);
    ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("/mo_cam/imu_info", 10);

    std::thread img_thread(publishIMGProcess, &hCameraHandle, std::move(img_pub), std::move(depth_pub));
    std::thread imu_thread(publishIMUProcess, &hCameraHandle, std::move(imu_pub));

    img_thread.join();
    imu_thread.join();

    moCloseCamera(&hCameraHandle);

    return 0;
}
