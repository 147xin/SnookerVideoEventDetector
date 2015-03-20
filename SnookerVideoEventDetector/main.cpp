//
//  main.cpp
//  SnookerVideoEventDetector
//
//  Created by Yixin Huang on 14/12/22.
//  Copyright (c) 2014年 Yixin Huang. All rights reserved.
//

#include "SnookerVideoEventDetector.h"

#include <fstream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>  // 使用了trim_copy函数

using namespace std;

int main(int argc, const char *argv[]) {
    /*
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
    */


    // --------------------------------------- 测试比分序列分析 ---------------------------------------
    // 首先获取比分序列
    SnookerVideoEventDetector detector;
    detector.videoInfo.fps = 30.0;
    detector.videoInfo.bestFrames = 7;
    ifstream ifile("/Users/hix/Desktop/SnookerVideoEventDetection/SnookerOCR/test_score_sequence.txt");
    string line, str;
    int num;
    while (getline(ifile, line)) {
//        istringstream lineStream(line);
//        //lineStream>>str>>str;
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
//        getline(lineStream,str,' ');
        FrameFeature record;
        record.frameId = stoi(boost::trim_copy(line.substr(6, 6)));
        record.name1 = boost::trim_copy(line.substr(14, 20));
        record.name2 = boost::trim_copy(line.substr(50, 20));
        record.score1 = stoi(boost::trim_copy(line.substr(37, 4)));
        record.score2 = stoi(boost::trim_copy(line.substr(43, 4)));
        record.frameScore1 = stoi(boost::trim_copy(line.substr(70, 4)));
        record.frameScore2 = stoi(boost::trim_copy(line.substr(74, 4)));
        record.bestFrames = stoi(boost::trim_copy(line.substr(78, 4)));

        string star1, star2;
        star1 = boost::trim_copy(line.substr(34, 3));
        star2 = boost::trim_copy(line.substr(47, 3));
        if (!star1.empty())
            record.turn = 0;
        else if (!star2.empty())
            record.turn = 1;

        detector.videoInfo.frameFeatures.push_back(record);
    }
    ifile.close();
    // 去除无效与重复记录
    detector.RefineScoreSequence();
    // 输出到文件中查看
//    ofstream ofile("/Users/hix/Desktop/SnookerVideoEventDetection/SnookerOCR/murphy_milkins比分序列_精炼.txt");
//    char buf[90];
//    vector<FrameFeature>::const_iterator it = detector.videoInfo.frameFeatures.cbegin();
//    for (; it != detector.videoInfo.frameFeatures.cend(); ++it) {
//        sprintf(buf, "Frame %6d: %20s%3s%4d  %-4d%3s%-20s%4d%4d%4d\n",
//                it->frameId,
//                it->name1.c_str(),
//                (it->turn == 0 ? " * " : ""),
//                it->score1,
//                it->score2,
//                (it->turn == 1 ? " * " : ""),
//                it->name2.c_str(),
//                it->frameScore1,
//                it->frameScore2,
//                it->bestFrames);
//        ofile << buf;
//    }
//    ofile.close();

    // 根据比分序列进行事件判定
    detector.EventDetection();
    return 0;
}
