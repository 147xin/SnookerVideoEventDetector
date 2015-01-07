#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
//#include "direct.h"
//extern "C"
//{
//#include "libavcodec/avcodec.h"
//#include "libavformat/avformat.h"
//#include "libswscale/swscale.h"
//}
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "CommonStructure.h"

using namespace std;

#define    FRAMEWIDTH    320
#define FRAMEHEIGHT    240

#define PI    3.1415926
//#define HISTBINS	16
#define LAMBDA_INFIELD    2.4//3.3	//1.3~2 1.3
#define LAMBDA_OUTFIELD 2.7//3.6 //1.3~2 1.4
#define THRESHOLD_N        3//2	//1~3
#define DOMINATCOLORTHRESHOLD1    0.3
#define DOMINATCOLORTHRESHOLD2    0.06
#define DOMINATCOLORTHRESHOLD3    0.2 //主色差大于此阈值则认为是不同镜头

#define DOMINATCOLOR_ALPHA        1.2    //1.0~1.5
#define DOMINATCOLOR_TD            20    //10~20

#define SLIDEWINDOWNUMBER        30    //滑动窗口大小35
#define GTWINDOWNUMBER            25//15	//渐变检测窗口
#define CANDIDACYGRADUAL        3    //候选渐变窗口

//主色提取范围阈值
#define  LOCALREGIONTHRESHOLD  0.2

#define Threshold 0.5
//
////直方图定义
//typedef struct HistData
//{
//	int h[HISTBINS];
//	int s[HISTBINS];
//	int v[HISTBINS];
//}HistData, *pHistData;

////渐变信息
//typedef struct _GTInfo
//{
//	int nStart;//开始位置
//	int nLength;//渐变长度
//}GTInfo;

class CShotCut {
public:
    CShotCut(void);

    ~CShotCut(void);

    int SetPath(char *imgpath, long long framecount);

    int ShotDetection(void);

    int SaveInfo(char *cutpath, char *gradientpath);

    vector<int> GetCutInfo(void);

    vector<GTInfo> GetGTInfo(void);

    long long GetFrameCount(void);

//	int ShotDetection_Old (char* oldcutpath, char* oldgradientpath);
private:
    //赋值直方图
    void CopyHistData(pHistData dstHist, pHistData srcHist);

    //计算两个直方图差的绝对值之和
    double calFrameHistDiff(pHistData prevHist, pHistData curHist);

    //主色提取（非mpeg-7标准） 参考张玉珍论文
    //输入参数为当前图像的颜色直方图
    CvScalar extractDominatColor(pHistData curHist);

    //计算主色率
    double calDominatColorRate(IplImage *pImg, pHistData curHist);

    double calDominatColorDiff(IplImage *pImg1, pHistData pHist1, IplImage *pImg2, pHistData pHist2);

private:
    int m_width;
    int m_height;
    char m_imgPath[260];
    //图片文件夹路径
    vector<int> m_CutInfo;
    //切变位置
    vector<GTInfo> m_GTInfo;
    //渐变信息
    long long m_nFrameCount;//视频帧数

};

