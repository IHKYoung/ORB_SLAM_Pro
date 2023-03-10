//
// Created by young on 22-12-1.
//
#include "common_function.h"
#include "opencv2/opencv.hpp"

#include "System.h"
#include <chrono>

// 处理ctrl+c中断
#include <csignal>

using namespace cv;
using namespace std;
using namespace std::chrono;

bool bEnd = false;

uint64_t timeSinceEpochMillisec()
{
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void signalHandler(int sig)
{
    if (sig == SIGINT)
    {
        bEnd = true;
        printf("\nORB_SLAM End and Camera Close!\n");
    }
}

int32_t main(int32_t argc, char **argv)
{

    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mo_rgbd_raw path_to_vocabulary path_to_settings path_of_video(default /dev/video0)"
             << endl;
        return 1;
    }

    signal(SIGINT, signalHandler);

    // Init ORB-SLAM3
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    float imageScale = SLAM.GetImageScale();

    printf("The version of SDK is %s\n", moGetSdkVersion());
    MO_CAMERA_HANDLE hCameraHandle = MO_INVALID_HANDLE;

    int32_t n32Result = 0;
    // Open camera by number
    // n32Result = moOpenUVCCameraByNumber(atoi(argv[3]), &hCameraHandle);
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
    uint8_t *pu8FrameBuffer = nullptr;
    // for RGBD
    uint16_t *pu16RGBDDisparityData = nullptr;
    uint8_t *pu8RGBDYUVI420Img = nullptr;

    // Set video mode to RGBD DENSE mode
    n32Result = moSetVideoMode(hCameraHandle, MVM_RGBD);
    if (0 != n32Result)
    {
        printf("Error: moSetVideoMode to RGBD(DENSE) failed %d\n", n32Result);
        return -3;
    }

    // Get the BxF(baseline * focus) and baseline of camera
    float pfBxf, pfBase; // pfBxf = Focus * pfBase
    n32Result = moGetBxfAndBase(hCameraHandle, &pfBxf, &pfBase);
    if (0 != n32Result)
    {
        printf("Error, moGetBxfAndBase Failed %d\n", n32Result);
        return -4;
    }
    printf("\nBxF = %f\nBaseline = %f\n", pfBxf, pfBase);

    printf("Press ENTER to quit the running process\n");

    while (0 == getchar_nb(FETCH_AND_DISPLAY_TIME_LENGTH))
    {
        // Get current video frame
        n32Result = moGetCurrentFrame(hCameraHandle, &u64ImageFrameNum, &pu8FrameBuffer);
        if (0 != n32Result)
        {
            printf("Get current frame failed. Try Again...\n");
            continue;
        }

        uint64_t nowTimestamp = timeSinceEpochMillisec();
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
        ///imshow("Disparity Show", matShowDisparity);
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
        //imshow("Depth Show", matDepth);

        SLAM.TrackRGBD(matYUV2BGR, matDepth, nowTimestamp);

        if (bEnd)
        {
            // Stop all threads
            SLAM.Shutdown();
            // Save camera trajectory
            time_t fileTime = system_clock::to_time_t(system_clock::now());
            std::stringstream fileTimeFormat;
            fileTimeFormat << std::put_time(std::localtime(&fileTime), "-%Y-%m-%d-%H-%M-%S");

            string fileTrajectorName = "moRawRGBDCameraTrajectory" + fileTimeFormat.str() + ".txt";
            string fileKFTrajectorName = "moRawRGBDKeyFrameTrajectory" + fileTimeFormat.str() + ".txt";

            SLAM.SaveTrajectoryMO(fileTrajectorName);
            SLAM.SaveKeyFrameTrajectoryMO(fileKFTrajectorName);
            break;
        }
    }

    // Close camera
    moCloseCamera(&hCameraHandle);
    return n32Result;
}
