//
//  SnookerVideoEventDetector.h
//  SnookerVideoEventDetector
//
//  Created by Yixin Huang on 14/12/22.
//  Copyright (c) 2014年 Yixin Huang. All rights reserved.
//

#ifndef SnookerVideoEventDetector_h
#define SnookerVideoEventDetector_h

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <cmath>
#include <cstring>
#include <boost/algorithm/string.hpp>  // 使用了trim_copy函数
#include <tesseract/baseapi.h>  // Google Tesseract OCR Engine
#include "ShotCut.h"
#include "ReplayDetector.h"

extern bool frameFeaturesDetectionStarted;

struct FrameFeature {
    int frameId = -1; //帧编号，初始值为-1
    bool hasTable = false;        //是否包含桌面
    bool isFullTable = false;     //是否是全台面
    bool hasScoreBar = false;     //是否包含比分条
    cv::Rect scorebarRegion;   //比分条在原帧中的位置, 这个字段用于获取候选的比分条区域
    //各字段在比分条中的位置
    int maxFrameNumPos[2] = {-1, -1};
    int frameNum1Pos[2] = {-1, -1};
    int frameNum2Pos[2] = {-1, -1};
    int score1Pos[2] = {-1, -1};
    int score2Pos[2] = {-1, -1};
    int name1Pos[2] = {-1, -1};
    int name2Pos[2] = {-1, -1};
    int name1PosPlanB[2] = {-1, -1};
    int name2PosPlanB[2] = {-1, -1};
    //比分条文本特征
    int turn = -1;                     //击球方 0/1，-1表示不确定
    std::string name1, name2;          //球员名字，获取候选球员名
    int score1 = -1, score2 = -1;           //当局比分
    int frameScore1 = -1, frameScore2 = -1; //大比分（局比分）
    int bestFrames;                        //总局数
};

enum AudioType {
    laughter, applause
};

struct AudioEvent {
    int startFrame;
    int length;
    AudioType audioType;
};

struct SnookerVideoInfo {
    std::string videoFilePath;               //视频文件路径
    std::string framesFolder;                //存放视频帧的文件夹
    std::string cutPath, gtPath, replayPath, replayFeatPath, allCutPath, cutUpdatePath,
            gtUpdatePath;                        //回放检测器需要用到的一些路径
    double fps;                              //帧率
    int framesNum;                  //总帧数
    int width, height;                       //视频尺寸
    std::string playerName1, playerName2;    //球员名字
    int bestFrames;                          //最大局数
    int bestFramesCharColor = -1;           //最大局数二值化后的字符（前景）颜色，0是黑色，255是白色，-1表示未确定
    cv::Rect scorebarRegion;                 //比分条在原帧中的位置
    int currentPlayerFlagPos[2][2];          //当前击球球员指示符在比分条中的位置
    std::vector<FrameFeature> frameFeatures; //取样帧的特征
    std::vector<ReplayInfo> replays;         //回放镜头
    std::unordered_set<int> replayCheckHashTab;
    std::vector<AudioEvent> audioEvents;     //音频事件
};


class SnookerVideoEventDetector {
public:

    SnookerVideoInfo videoInfo;
    std::vector<cv::Rect> candidateScorebarRegions;
    cv::Mat lastScorebar, currentScorebar;
    std::string extendedPlayerListPath;

    void SetExtendedPlayerListPath(const std::string &path) {
        extendedPlayerListPath = path;
    }

    SnookerVideoEventDetector(); //构造函数
    void SetVideoFilePath(const std::string &videoPath);

    void GetVideoFrames(const std::string &framesPath);

    void SetReplayDetectorOutputPath(const std::string &outputPath);

    void GetReplayInfo();

    void ReadReplayInfo();

    void GetFrameFeature(int const currentFrameNum, FrameFeature &frameFeature);

    void GetScorebarRegion();

    bool GetCurrentPlayerFlagPos();

    void GetVideoFramesFeature();


private:
    void DrawDetectedLines(cv::Mat &image, const std::vector<cv::Vec2f> &lines,
                           const cv::Scalar &color, const int width = 1);

    void DrawDetectedLinesP(cv::Mat &image, const std::vector<cv::Vec4i> &lines, cv::Scalar &color);

    void DrawCircles(cv::Mat &image, const std::vector<cv::Point> &points, const cv::Scalar &color);

    void IntersectionPoint(const cv::Vec2f line1, const cv::Vec2f line2,
                           std::vector<cv::Point> &points, int imgRows, int imgCols);

    void SelectPoint(const std::vector<cv::Point> &points, cv::Size imageSize,
                     cv::Point &selectedPoint, bool &leftSide);

    int TableRegionHeight(const std::vector<cv::Point> &points, const int imageHeight);

    bool IsFullTableView(const std::vector<cv::Vec2f> &lines, const cv::Size imageSize);

    void RemoveNearbyPoints(std::vector<cv::Point> &points, const cv::Size imageSize);

    void RemoveNearbyLines(std::vector<cv::Vec2f> &lines, const cv::Size imageSize);

    void RemoveSmallObjects(cv::Mat &image, const cv::Size &imageSize);

    void GetGrayScorebarFromFrameId(int const frameId, cv::Mat &scorebar);

    bool DetectTurnIndicator(cv::Mat &scorebarDiff);

    void GetCorrectNames(const std::string &name1, const std::string &name2);

    int EditDistance(const std::string &str1, const std::string &str2);
};

#endif
