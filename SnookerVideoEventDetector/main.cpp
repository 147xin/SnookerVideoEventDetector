//
//  main.cpp
//  SnookerVideoEventDetector
//
//  Created by Yixin Huang on 14/12/22.
//  Copyright (c) 2014年 Yixin Huang. All rights reserved.
//

#include "SnookerVideoEventDetector.h"

int main(int argc, const char *argv[]) {
    SnookerVideoEventDetector detector;
    detector.SetVideoFilePath("/Users/hix/Desktop/SnookerVideoEventDetection/"
                                      "SnookerVideos/Murphy147/");

    //detector.GetVideoFrames("/Users/hix/Desktop/SnookerVideoEventDetection/"
    //                        "TestReplayDetection/Frames1/");
    //为节省测试时间, 这里不再重复获取视频帧, 直接设置帧图像文件夹、帧率和总帧数、视频尺寸
    detector.videoInfo.framesFolder = "/Users/hix/Desktop/SnookerVideoEventDetection/TestReplayDetection/Frames1/";
    detector.videoInfo.fps = 30;
    detector.videoInfo.framesNum = 19772;
    detector.videoInfo.width = 1105;
    detector.videoInfo.height = 622;

    detector.SetReplayDetectorOutputPath("/Users/hix/Desktop/"
                                                 "SnookerVideoEventDetection/"
                                                 "TestReplayDetection/Output/");
    // 设置扩展球员名列表文件路径
    detector.SetExtendedPlayerListPath("/Users/hix/PycharmProjects/SnookerPlayerNameListExtractor"
                                               "/extended_player_list.txt");
    //detector.GetReplayInfo(); //节省调试时间, 暂时注释掉
    //读取回放信息
    detector.ReadReplayInfo();

    //获取比分条位置
    detector.GetScorebarRegion();
    //获取当前击球球员指示符位置
    detector.GetCurrentPlayerFlagPos();
    //获取每一样本帧的特征
    extern bool frameFeaturesDetectionStarted;
    frameFeaturesDetectionStarted = true;
    detector.GetVideoFramesFeature();

    return 0;
}
