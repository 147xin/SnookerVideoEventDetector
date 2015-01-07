#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
//#include <algorithm >
//#include "direct.h"
#include <time.h>
//extern "C"
//{
//#include "libavcodec/avcodec.h"
//#include "libavformat/avformat.h"
//#include "libswscale/swscale.h"
//}
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/legacy/legacy.hpp> //added by hyx in 2014-04-18
#include "CommonStructure.h"
#include <iostream>

using namespace std;

const int MAXFRAMECOUNT = 30;
//20
const int MAXBLOCKCOUNT = 8;
const int IMAGEWIDTH = 320;
const int IMAGEHEIGHT = 240;
const int nWinSize = 40;
const double ZERO = 0.000001;
const double pi = 3.1415926;

//块光流场
typedef struct _blockFlow {
    double xvalue;
    double yvalue;
    int direction;
    int id;
} BlockFlow, *PBlockFlow;

////渐变信息
//typedef struct _GTInfo
//{
//	int nStart;//开始位置
//	int nLength;//渐变长度
//}GTInfo;

typedef double **SCSEQUENES;//两序列帧间相似度

class SumPoint {
public:
    //SumPoint(long sum, int x, int y):value(sum),i(x),j(y){}
    float value;
    int x;
    int y;
};

//降序排列
extern bool greater_value(const SumPoint &c1, const SumPoint &c2);

//升序排列
extern bool less_value(const SumPoint &c1, const SumPoint &c2);

//降序排列
extern bool greater_coordinate(const SumPoint &c1, const SumPoint &c2);

//升序排列
extern bool less_coordinate(const SumPoint &c1, const SumPoint &c2);

//回放镜头检测类
class CReplayDetector {
public:
    CReplayDetector(void);

    ~CReplayDetector(void);

public:
    int SetPath(char *gradientpath, char *cutpath, char *imgpath);

    //设置渐变文件路径和图片文件夹路径
    int SetVideoFrames(int framecount);

    //设置视频总帧数
    int ReadFile(void);

    //读取渐变信息
    int CalcOpticalFlow(void);

    int GetLogoTemplate(void);

    int GetLpixels(void);

    int LogoDetection(void);

    int UpdateShotCutInfo(void);

    int SaveInfo(char *replaypath, char *replayFeatpath, char *allcutpath, char *gradientpath = NULL, char *cutpath = NULL);

    bool isReplayExist();

private:
    int ImgOpticalFlow(IplImage *prev_grey, IplImage *grey, int nWinSize, BlockFlow *dbBlockFlow);

    double SC_frames(BlockFlow *frame1, BlockFlow *frame2);

    double max3(double a, double b, double c);

    double Alignment(int firstPos, int secondPos, int nSeqLen1, int nSeqLen2);

    int CalcDirection(double xx, double yy);

    double VerifyLogoSeq(GTInfo info);

private:
    char m_gradientPath[260];
    //渐变信息文件
    char m_cutPath[260];        //切变信息文件
    char m_imgPath[260];        //图片文件夹路径
    double m_dbAvgScore;        //平均序列匹配值 大于此值，认为序列匹配成功
    vector<GTInfo> m_vecGTInfo;        //渐变信息(经过过滤,LOGO检测使用)
    vector<GTInfo> m_vecAllGTInfo;        //渐变信息(所有，包括误检部分，以及某些漏检，最后Replay匹配时用来过滤误检渐变)
    vector<GTInfo> m_vecUpdateGTInfo;    //修正后的渐变信息
    vector<int> m_vecCutInfo;        //切变信息
    vector<int> m_vecUpdateCutInfo;    //修正后的切变信息
    vector<PBlockFlow *> m_SeqFlow;            //序列top光流
    int *m_logoFlag;            //序列是logo序列的标志
    int m_logoCount;        //序列是logo序列的个数
    SCSEQUENES **m_scFrame;            //各序列帧之间相似度
    int m_nMaxSeqPos;        //最大序列比对渐变 值为m_vecGTInfo中下标
    int m_nKframePos;        //Kframe在最大序列比对渐变中的位置
    int *m_RframePos;        //logo序列中R-frame的位置
    int m_nVideoFrames;        //视频总帧数
    IplImage *m_pMaskImage;        //L-pixels图像
    int m_nLpixelsCount;    //L-pixels像素个数
    vector<int> m_confirmLogo;        //确定的logo序列起点
    vector<ReplayInfo> m_vecReplay;        //replay信息
    vector<int> m_vecAllCut;        //所有镜头变换点
    clock_t start, finish;        //计时参数
};

