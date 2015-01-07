//#include "StdAfx.h"
#include "ShotCut.h"
#include <time.h>

//本文件主要对视频镜头的切变和渐变进行了检测
CShotCut::CShotCut(void) {
    m_nFrameCount = 0;
}


CShotCut::~CShotCut(void) {
    vector<int>().swap(m_CutInfo);
    vector<GTInfo>().swap(m_GTInfo);
}

int CShotCut::SetPath(char *imgpath, long long framecount) {
    if (imgpath == NULL || framecount <= 0) {
        return 0;
    }

    strcpy(m_imgPath, imgpath);
    m_nFrameCount = framecount;
    return 1;
}

//赋值直方图
void CShotCut::CopyHistData(pHistData dstHist, pHistData srcHist) {
    for (int i = 0; i < HISTBINS; i++) {
        dstHist->h[i] = srcHist->h[i];
        dstHist->s[i] = srcHist->s[i];
        dstHist->v[i] = srcHist->v[i];
    }
}

//计算两个直方图差的绝对值之和
double CShotCut::calFrameHistDiff(pHistData prevHist, pHistData curHist) {
    double frameDiff = 0.0;

    for (int i = 0; i < HISTBINS; i++) {
        frameDiff += abs(prevHist->h[i] - curHist->h[i]);
    }
    for (int i = 0; i < HISTBINS; i++) {
        frameDiff += abs(prevHist->s[i] - curHist->s[i]);
    }
    for (int i = 0; i < HISTBINS; i++) {
        frameDiff += abs(prevHist->v[i] - curHist->v[i]);
    }

    return frameDiff / (m_width * m_height);
}


//主色提取（非mpeg-7标准） 参考张玉珍论文
//输入参数为当前图像的颜色直方图
CvScalar CShotCut::extractDominatColor(pHistData curHist) {
    int maxPosH = 0, maxPosS = 0, maxPosV = 0;
    int maxValueH = 0, maxValueS = 0, maxValueV = 0;
    int leftPosH = 0, rightPosH = 0;
    int leftPosS = 0, rightPosS = 0;
    int leftPosV = 0, rightPosV = 0;
    double meanPosH = 0, meanPosS = 0, meanPosV = 0;
    double meanValueH = 0, meanValueS = 0, meanValueV = 0;
    CvScalar cDominatMean;
    cDominatMean.val[0] = 0;
    cDominatMean.val[1] = 0;
    cDominatMean.val[2] = 0;
    cDominatMean.val[3] = 0;

    //查找峰值
    for (int i = 0; i < HISTBINS; i++) {
        if (curHist->h[i] > maxValueH) {
            maxValueH = curHist->h[i];
            maxPosH = i;
        }
        if (curHist->s[i] > maxValueS) {
            maxValueS = curHist->s[i];
            maxPosS = i;
        }
        if (curHist->v[i] > maxValueV) {
            maxValueV = curHist->v[i];
            maxPosV = i;
        }
    }

    //区间范围H
    leftPosH = rightPosH = maxPosH;
    for (int i = maxPosH - 1; i >= 0; i--) {
        if (curHist->h[i] < maxValueH * LOCALREGIONTHRESHOLD) {
            break;
        }
        else {
            leftPosH = i;
        }
    }
    for (int i = maxPosH + 1; i < HISTBINS; i++) {
        if (curHist->h[i] < maxValueH * LOCALREGIONTHRESHOLD) {

            break;
        }
        else {
            rightPosH = i;
        }
    }
    //区间范围S
    leftPosS = rightPosS = maxPosS;
    for (int i = maxPosS - 1; i >= 0; i--) {
        if (curHist->s[i] < maxValueS * LOCALREGIONTHRESHOLD) {
            break;
        }
        else {
            leftPosS = i;
        }
    }
    for (int i = maxPosS + 1; i < HISTBINS; i++) {
        if (curHist->s[i] < maxValueS * LOCALREGIONTHRESHOLD) {
            break;
        }
        else {
            rightPosS = i;
        }
    }
    //区间范围V
    leftPosV = rightPosV = maxPosV;
    for (int i = maxPosV - 1; i >= 0; i--) {
        if (curHist->v[i] < maxValueV * LOCALREGIONTHRESHOLD) {
            break;
        }
        else {
            leftPosV = i;
        }
    }
    for (int i = maxPosV + 1; i < HISTBINS; i++) {
        if (curHist->v[i] < maxValueV * LOCALREGIONTHRESHOLD) {
            break;
        }
        else {
            rightPosV = i;
        }
    }

    //计算区间范围内的均值
    float nTotal = 0;
    float nTotalMult = 0;
    for (int i = leftPosH; i <= rightPosH; i++) {
        nTotal += curHist->h[i];
        nTotalMult += curHist->h[i] * i;
    }
    meanPosH = nTotalMult / nTotal;
    meanValueH = meanPosH * 180 / HISTBINS;

    nTotal = 0;
    nTotalMult = 0;
    for (int i = leftPosS; i <= rightPosS; i++) {
        nTotal += curHist->s[i];
        nTotalMult += curHist->s[i] * i;
    }
    meanPosS = nTotalMult / nTotal;
    meanValueS = meanPosS * 255 / HISTBINS;

    nTotal = 0;
    nTotalMult = 0;
    for (int i = leftPosV; i <= rightPosV; i++) {
        nTotal += curHist->v[i];
        nTotalMult += curHist->v[i] * i;
    }
    meanPosV = nTotalMult / nTotal;
    meanValueV = meanPosV * 255 / HISTBINS;

    //主色结构
    cDominatMean.val[0] = meanValueH;
    cDominatMean.val[1] = meanValueS;
    cDominatMean.val[2] = meanValueV;

    return cDominatMean;
}

//计算主色率
double CShotCut::calDominatColorRate(IplImage *pImg, pHistData curHist) {
    //IplImage *pDominatImg = cvCreateImage(cvGetSize(pImg), 8, 1);
    CvScalar s = cvScalar(0);
    CvScalar cMeanColor = cvScalar(0);
    double distance = 0;
    double dSinHmean = 0;
    double dCosHmean = 0;
    double dSinHxy = 0, dCosHxy = 0;
    int nCount = 0;

    //cvZero( pDominatImg );
    cMeanColor = extractDominatColor(curHist);

    dSinHmean = sin(cMeanColor.val[0] * PI / 180.0);
    dCosHmean = cos(cMeanColor.val[0] * PI / 180.0);

    for (int i = 0; i < pImg->height; i++) {
        for (int j = 0; j < pImg->width; j++) {
            //获取（i,j）像素点
            s = cvGet2D(pImg, i, j);

            dSinHxy = sin(s.val[0] * PI / 180.0);
            dCosHxy = cos(s.val[0] * PI / 180.0);

            //计算距离
            distance = sqrt((s.val[2] - cMeanColor.val[2]) * (s.val[2] - cMeanColor.val[2])
                    + (s.val[1] * dCosHxy - cMeanColor.val[1] * dCosHmean) * (s.val[1] * dCosHxy - cMeanColor.val[1] * dCosHmean)
                    + (s.val[1] * dSinHxy - cMeanColor.val[1] * dSinHmean) * (s.val[1] * dSinHxy - cMeanColor.val[1] * dSinHmean));

            //85 经验阈值
            if (distance < 90) {
                //cvSet2D( pDominatImg, i, j, cvScalar(255) );
                nCount++;
            }
        }
    }
    //cvNamedWindow("Image",1);
    //cvShowImage("Image",pImg);
    //cvNamedWindow("Dominat",1);
    //cvShowImage("Dominat",pDominatImg);
    //cvWaitKey(0); //等待按键
    //cvDestroyWindow( "Image" );//销毁窗口
    //cvReleaseImage( &pDominatImg ); //释放图像

    return (double) nCount / (pImg->height * pImg->width);
}

double CShotCut::calDominatColorDiff(IplImage *pImg1, pHistData pHist1, IplImage *pImg2, pHistData pHist2) {
    double dRateDiff = 0.0;
    CvScalar cMeanColor1 = extractDominatColor(pHist1);
    CvScalar cMeanColor2 = extractDominatColor(pHist2);
    double dSinHmean1 = sin(cMeanColor1.val[0] * PI / 180.0);
    double dCosHmean1 = cos(cMeanColor1.val[0] * PI / 180.0);
    double dSinHmean2 = sin(cMeanColor2.val[0] * PI / 180.0);
    double dCosHmean2 = cos(cMeanColor2.val[0] * PI / 180.0);

    //计算主色距离
    double distance = sqrt((cMeanColor1.val[2] - cMeanColor2.val[2]) * (cMeanColor1.val[2] - cMeanColor2.val[2])
            + (cMeanColor1.val[1] * dCosHmean1 - cMeanColor2.val[1] * dCosHmean2) * (cMeanColor1.val[1] * dCosHmean1 - cMeanColor2.val[1] * dCosHmean2)
            + (cMeanColor1.val[1] * dSinHmean1 - cMeanColor2.val[1] * dSinHmean2) * (cMeanColor1.val[1] * dSinHmean1 - cMeanColor2.val[1] * dSinHmean2));

    //先判断主色颜色差别，再判断主色率差别
    //如果主色颜色差别比较大，就没必要判断主色率
    if (distance < 25) {
        double dRate1 = calDominatColorRate(pImg1, pHist1);
        double dRate2 = calDominatColorRate(pImg2, pHist2);
        dRateDiff = abs(dRate1 - dRate2);
    }
    else {
        dRateDiff = 0.5;
    }

    return dRateDiff;
}

/************************************************************************
函数名称：ShotDetection
函数作用：检测视频中的渐变和切变。渐变保存在m_GTInfo中，切变保存在m_CutInfo中。
参数：无
返回值：
*************************************************************************/
int CShotCut::ShotDetection(void) {
    //计时参数
    clock_t start, finish;
    double totaltime;
    start = clock();
    //fprintf(stderr, "镜头分割进行中，请稍候...\n");
    cout << "Shot Detecting..." << endl;

    // 读入原始图像
    IplImage *pImage = NULL;
    IplImage *hsv = NULL;
    IplImage *h_plane = NULL;
    IplImage *s_plane = NULL;
    IplImage *v_plane = NULL;
    //H分量划分为16个等级，S分量划分为16个等级, V分量划分为16个等级
    int h_bins = HISTBINS, s_bins = HISTBINS, v_bins = HISTBINS;
    //H分量的变化范围
    float h_ranges_arr[] = {0, 180};
    float *h_ranges = h_ranges_arr;
    //S分量的变化范围
    float s_ranges_arr[] = {0, 255};
    float *s_ranges = s_ranges_arr;
    //V分量的变化范围
    float v_ranges_arr[] = {0, 255};
    float *v_ranges = v_ranges_arr;
    //创建直方图 H维
    CvHistogram *h_hist = cvCreateHist(1, &h_bins, CV_HIST_ARRAY, &h_ranges, 1);
    //创建直方图 S维
    CvHistogram *s_hist = cvCreateHist(1, &s_bins, CV_HIST_ARRAY, &s_ranges, 1);
    //创建直方图 V维
    CvHistogram *v_hist = cvCreateHist(1, &v_bins, CV_HIST_ARRAY, &v_ranges, 1);
    //保存帧直方图数据
    vector<pHistData> vecHistData;//HistData指针，暂时未使用
    //上一帧图像数据，用于计算主色率差
    IplImage *prevImage = NULL;
    //上一帧直方图数据
    pHistData prevHistData = (pHistData) malloc(sizeof(HistData));
    //当前帧直方图数据
    pHistData currHistData = (pHistData) malloc(sizeof(HistData));
    //渐变起始帧直方图数据，并分配空间
    pHistData histGTStart = (pHistData) malloc(sizeof(HistData));
    //保存滑动窗口的帧间差
    double vecSlideWinDiff[SLIDEWINDOWNUMBER] = {0.0};
    //渐变检测过程中保存的帧间差
    double vecGTDiff[GTWINDOWNUMBER] = {0.0};
    //滑动窗口
    int nSlideWindow = 0;
    //可能渐变标记
    bool bGT = false;
    //渐变类型，为0是不包含足球场地，为1是包含足球场地
    int nGTType = 0;
    //渐变检测中的相隔帧差（单调增检测）
//	double dbGTDiff = 0.0;
    //帧差大于最小阈值的帧数
    int nDiffCount = 0;
    //渐变最大帧差 10
    int nMaxGTCount = 0;
    //渐变开始图像帧
    IplImage *pGTImage = NULL;
    //渐变开始帧号
    int nGTStartNum = 0;
    //渐变结束帧号
    int nGTEndNum = 0;
    //可能切变 如果后一帧的帧差小于最小阈值则认定为切变，否则认为是候选渐变的起始帧
    bool bCandidacyCut = false;
    //滑动窗口是否充足
    bool bSlideEnough = false;
    //上一帧差
    double dbLastDiff = 0.0;
    //是否是新一个镜头的开始 某些切变过渡帧为两帧 dbLastDiff和bIsNewCut处理这种情况
    bool bIsNewCut = false;

    //双阈值
    double dThresholdHigh1 = 0.0;
    double dThresholdLow1 = 0.0;
    double dThresholdHigh2 = 0.0;
    double dThresholdLow2 = 0.0;

    //文件路径
    char szPicName[256] = {0};

    for (long long i = 0; i < m_nFrameCount; i++) {
        sprintf(szPicName, "%s/frame%lld.jpg", m_imgPath, i);
        pImage = cvLoadImage(szPicName, 1);
        if (i == 0) {//初始化前一帧和渐变图像
            m_width = pImage->width;
            m_height = pImage->height;
            prevImage = cvCreateImage(cvSize(pImage->width, pImage->height), pImage->depth, pImage->nChannels);
            pGTImage = cvCreateImage(cvSize(pImage->width, pImage->height), pImage->depth, pImage->nChannels);
        }
        //直方图计算
        if (!hsv) {
            hsv = cvCreateImage(cvGetSize(pImage), 8, 3);
            h_plane = cvCreateImage(cvGetSize(pImage), 8, 1);
            s_plane = cvCreateImage(cvGetSize(pImage), 8, 1);
            v_plane = cvCreateImage(cvGetSize(pImage), 8, 1);
        }

        /** 输入图像转换到HSV颜色空间 */
        cvCvtColor(pImage, hsv, CV_BGR2HSV);
        cvCvtPixToPlane(hsv, h_plane, s_plane, v_plane, 0);
        /** 计算直方图 H维 */
        cvCalcHist(&h_plane, h_hist, 0, 0);
        /** 计算直方图 S维 */
        cvCalcHist(&s_plane, s_hist, 0, 0);
        /** 计算直方图 V维 */
        cvCalcHist(&v_plane, v_hist, 0, 0);

        //获得直方图中的统计次数
        for (int k = 0; k < h_bins; k++) {
            currHistData->h[k] = cvQueryHistValue_1D(h_hist, k);
        }
        for (int k = 0; k < s_bins; k++) {
            currHistData->s[k] = cvQueryHistValue_1D(s_hist, k);
        }
        for (int k = 0; k < v_bins; k++) {
            currHistData->v[k] = cvQueryHistValue_1D(v_hist, k);
        }

        if (i == 0) {//视频第一帧,存储直方图数据

            //复制到上一帧直方图变量
            CopyHistData(prevHistData, currHistData);
            ////复制第一帧
            //cvCopyImage(pImage, prevImage);
            //cvReleaseImage( &pImage );
        }
        else {
            //计算帧间差
            double curDiff = calFrameHistDiff(prevHistData, currHistData);

            ////保存帧间差
            //fresult<<i<<","<<curDiff<<","<<dThresholdHigh1<<","<<dThresholdLow1<<endl;

            //滑动窗口未填充足够，即两个镜头不会离得太近
            if (nSlideWindow < 16) {
                if (nSlideWindow == 0 && !bIsNewCut) {
                    if (curDiff < dbLastDiff / 3) {
                        memmove(&vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]));
                        vecSlideWinDiff[0] = curDiff;
                        nSlideWindow++;
                        CopyHistData(prevHistData, currHistData);
                        bSlideEnough = false;
                    }

                    bIsNewCut = true;
                }
                else {
                    memmove(&vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]));
                    vecSlideWinDiff[0] = curDiff;
                    nSlideWindow++;
                    CopyHistData(prevHistData, currHistData);
                    bSlideEnough = false;
                }
            }
            else {
                bSlideEnough = true;
            }

            dbLastDiff = curDiff;

            //判断候选切变是否为切变，如果不是切变，则认为是候选渐变
            if (bCandidacyCut && bSlideEnough) {
                if (curDiff < dThresholdLow2 * 2) {//帧差小于低阈值 认为候选切变是切变
                    vecSlideWinDiff[0] = curDiff;
                    nSlideWindow = 1;
                    m_CutInfo.push_back(i - 1);

                    bIsNewCut = false;
                }
                else {//帧差大于低阈值 认为是候选渐变
                    bGT = true;
                    nGTType = 0;
                    nDiffCount = 1;
                }
                bCandidacyCut = false;
            }

            //如果不是可能渐变的开始，检测帧间差
            if (bGT == false && bSlideEnough) {
                //计算滑动窗口均值和标准差
                double sum = 0, mean = 0;
                double variance = 0, stddev = 0;

                if (nSlideWindow == 0) {
                    mean = curDiff;
                    stddev = 0;
                }
                else {
                    for (int nsize = 0; nsize < nSlideWindow; nsize++) {
                        sum += vecSlideWinDiff[nsize];
                    }
                    mean = sum / nSlideWindow;
                    for (int nsize = 0; nsize < nSlideWindow; nsize++) {
                        variance += (vecSlideWinDiff[nsize] - mean) * (vecSlideWinDiff[nsize] - mean);
                    }
                    stddev = sqrt(variance / nSlideWindow);
                }
                //计算高低双阈值
                dThresholdHigh1 = (mean + THRESHOLD_N * stddev) * LAMBDA_OUTFIELD;
                dThresholdLow1 = mean * LAMBDA_OUTFIELD;
                dThresholdHigh2 = (mean + THRESHOLD_N * stddev) * LAMBDA_INFIELD;
                dThresholdLow2 = mean * LAMBDA_INFIELD;

                //fthreshold1<<i<<","<<dThresholdHigh1<<endl;
                //fthreshold2<<i<<","<<dThresholdLow1<<endl;

                //当前帧差大于最小低阈值
                if (curDiff > dThresholdLow2) {
                    if (calDominatColorRate(pImage, currHistData) <= DOMINATCOLORTHRESHOLD1) {//不包含足球场地
                        double dDomColorDiff = calDominatColorDiff(pImage, currHistData, prevImage, prevHistData);

                        if ((curDiff > dThresholdHigh1 * 2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh1 * 3) /*|| dDomColorDiff > 0.3*/ )//加入dDomColorDiff > 0.3切变准确率高，渐变检测结果少些
                        {
                            //发生突变
                            nSlideWindow = 0;
                            //记录突变
                            m_CutInfo.push_back(i);
                            bIsNewCut = false;
                        }
                        else if ((curDiff > dThresholdHigh1 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh1 * 2)) {
                            //候选突变
                            bCandidacyCut = true;
                            //保存渐变起始帧直方图和起始相隔帧差
                            CopyHistData(histGTStart, prevHistData);
                            cvCopyImage(prevImage, pGTImage);
                            vecGTDiff[GTWINDOWNUMBER - 1] = curDiff;
                            nGTStartNum = i;
                        }
                        else if (curDiff <= dThresholdHigh1 && curDiff > dThresholdLow1
                            /*&& dDomColorDiff > DOMINATCOLORTHRESHOLD2*/ ) {
                            //可能存在渐变的起始帧
                            bGT = true;
                            nGTType = 0;
                            nDiffCount++;
                            //保存渐变起始帧直方图和起始相隔帧差
                            CopyHistData(histGTStart, prevHistData);
                            cvCopyImage(prevImage, pGTImage);
                            vecGTDiff[GTWINDOWNUMBER - 1] = curDiff;
                            nGTStartNum = i;
                        }
                        else {//不是镜头边界，继续增大滑动窗口
                            //保存滑动窗口帧间差
                            memmove(&vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]));
                            vecSlideWinDiff[0] = curDiff;
                            if (nSlideWindow < SLIDEWINDOWNUMBER)
                                nSlideWindow++;
                        }
                    }
                    else {//包含足球场地
                        double dDomColorDiff = calDominatColorDiff(pImage, currHistData, prevImage, prevHistData);
                        if ((curDiff > dThresholdHigh2 * 2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh2 * 3) /*|| dDomColorDiff > 0.3*/ )//加入dDomColorDiff > 0.3切变准确率高，渐变检测结果少些
                        {
                            //发生突变
                            nSlideWindow = 0;
                            //记录突变
                            m_CutInfo.push_back(i);
                            bIsNewCut = false;
                        }
                        else if ((curDiff > dThresholdHigh2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh2 * 2)) {
                            //候选突变
                            bCandidacyCut = true;
                            //保存渐变起始帧直方图和起始相隔帧差
                            CopyHistData(histGTStart, prevHistData);
                            cvCopyImage(prevImage, pGTImage);
                            vecGTDiff[GTWINDOWNUMBER - 1] = curDiff;
                            nGTStartNum = i;
                        }
                        else if (curDiff <= dThresholdHigh2 && curDiff > dThresholdLow2
                            /*&& dDomColorDiff > DOMINATCOLORTHRESHOLD2*/ ) {
                            //可能存在渐变的起始帧
                            bGT = true;
                            nGTType = 1;
                            nDiffCount++;
                            //保存渐变起始帧直方图和起始相隔帧差
                            CopyHistData(histGTStart, prevHistData);
                            cvCopyImage(prevImage, pGTImage);
                            vecGTDiff[GTWINDOWNUMBER - 1] = curDiff;
                            nGTStartNum = i;
                        }
                        else {//不是镜头边界，继续增大滑动窗口
                            //保存滑动窗口帧间差
                            memmove(&vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]));
                            vecSlideWinDiff[0] = curDiff;
                            if (nSlideWindow < SLIDEWINDOWNUMBER)
                                nSlideWindow++;
                        }
                    }
                }
                else {
                    //如果不是渐变检测状态则帧间差存入滑动窗口,如果是渐变在渐变检测中处理
                    //保存滑动窗口帧间差
                    memmove(&vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]));
                    vecSlideWinDiff[0] = curDiff;
                    if (nSlideWindow < SLIDEWINDOWNUMBER)
                        nSlideWindow++;
                }

            }
            else if (bSlideEnough) {//正在渐变检测过程中
                if (nMaxGTCount < GTWINDOWNUMBER - 1) {
                    nMaxGTCount++;
                    vecGTDiff[GTWINDOWNUMBER - nMaxGTCount - 1] = curDiff;
                    if (nSlideWindow < SLIDEWINDOWNUMBER)
                        nSlideWindow++;
                    //如果当前帧差大于最大帧差的3倍，则认为是突变
                    if (curDiff > dThresholdHigh1 * 4/*(nGTType==0 ? dThresholdHigh1*3 : dThresholdHigh2*3)*/ ) {
                        //终止渐变检测，是突变
                        bGT = false;
                        nMaxGTCount = 0;
                        nDiffCount = 0;
                        nSlideWindow = 0;
                        //记录突变
                        m_CutInfo.push_back(i);
                        bIsNewCut = false;
                    }
                    else if (curDiff > (nGTType == 0 ? dThresholdLow1 : dThresholdLow2)) {
                        nDiffCount++;
                        nGTEndNum = i;
                    }
                    //渐变特征 前三个渐变帧中至少有两个帧间差大于最低阈值
                    //如果不满足这个特征，取消此次渐变检测，并将渐变帧差添加入滑动窗口
                    if (nMaxGTCount == CANDIDACYGRADUAL) {
                        if (nDiffCount < 2) {
                            //终止渐变检测，不是候选渐变
                            bGT = false;
                            memmove(&vecSlideWinDiff[CANDIDACYGRADUAL], &vecSlideWinDiff[0],
                                    sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) * CANDIDACYGRADUAL);
                            memmove(&vecSlideWinDiff[0], &vecGTDiff[GTWINDOWNUMBER - CANDIDACYGRADUAL], sizeof(vecGTDiff[0]) * CANDIDACYGRADUAL);
                            nMaxGTCount = 0;
                            nDiffCount = 0;
                            //判断候选渐变中是否存在切变
                            for (int k = 1; k < CANDIDACYGRADUAL; k++) {
                                if (vecGTDiff[GTWINDOWNUMBER - k] > dThresholdHigh2) {
                                    nSlideWindow = GTWINDOWNUMBER - k - 1;
                                    //记录突变
                                    m_CutInfo.push_back(nGTStartNum + k - 1);
                                    bIsNewCut = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                else {
                    //相隔帧超过阈值
                    double dBeforeDomColorDiff = calDominatColorDiff(pImage, currHistData, pGTImage, histGTStart);
                    if (calFrameHistDiff(histGTStart, currHistData) > (nGTType == 0 ? dThresholdHigh1 : dThresholdHigh2) &&
                            dBeforeDomColorDiff > DOMINATCOLORTHRESHOLD2 &&
                            nDiffCount > 3) {
                        //渐变终止帧，检测出渐变
                        bGT = false;
                        GTInfo tmpInfo;
                        tmpInfo.nStart = nGTStartNum;
                        tmpInfo.nLength = nGTEndNum - nGTStartNum + 1;
                        m_GTInfo.push_back(tmpInfo);
                        bIsNewCut = false;

                        //镜头开始，初始化滑动窗口
                        nSlideWindow = i - nGTEndNum;
                        memmove(&vecSlideWinDiff[0], &vecGTDiff[0], sizeof(vecGTDiff[0]) * nSlideWindow);
                        nMaxGTCount = 0;
                        nDiffCount = 0;

                    }
                    else {
                        bGT = false;
                        memmove(&vecSlideWinDiff[GTWINDOWNUMBER], &vecSlideWinDiff[0],
                                sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) * GTWINDOWNUMBER);
                        memmove(&vecSlideWinDiff[0], &vecGTDiff[0], sizeof(vecGTDiff));
                        nMaxGTCount = 0;
                        nDiffCount = 0;
                        //判断候选渐变中是否存在切变 20100929 11:28
                        for (int k = 1; k < GTWINDOWNUMBER; k++) {
                            if (vecGTDiff[GTWINDOWNUMBER - k] > dThresholdHigh2/**2*/ ) {
                                //发生突变
                                nSlideWindow = GTWINDOWNUMBER - k;
                                //记录突变
                                m_CutInfo.push_back(i - nSlideWindow - 1);
                                bIsNewCut = false;
                                break;
                            }
                        }
                    }
                }
            }

            //释放上一个直方图数据，重新赋值
            CopyHistData(prevHistData, currHistData);
        }
        cvCopyImage(pImage, prevImage);
        cvReleaseImage(&pImage);
    }

    //最后一帧算是镜头结尾
    m_CutInfo.push_back(m_nFrameCount - 1);

    //cvReleaseImage( &pImage );
    cvReleaseImage(&prevImage);
    cvReleaseImage(&pGTImage);
    cvReleaseImage(&hsv);
    cvReleaseImage(&h_plane);
    cvReleaseImage(&s_plane);
    cvReleaseImage(&v_plane);
    free(prevHistData);
    free(currHistData);
    free(histGTStart);
    cvReleaseHist(&h_hist);
    cvReleaseHist(&s_hist);
    cvReleaseHist(&v_hist);

    //fresult.close();
    //fthreshold1.close();
    //fthreshold2.close();

    finish = clock();
    totaltime = (double) (finish - start) / CLOCKS_PER_SEC;
    //fprintf(stderr, "镜头分割完成！共耗时%.02lf秒！\n", totaltime );
    cout << "Shot Detection Over, Use:" << totaltime << " seconds!" << endl;
    return 1;
}

int    CShotCut::SaveInfo(char *cutpath, char *gradientpath) {
    if (cutpath == NULL || gradientpath == NULL) {
        return 0;
    }
    FILE *fp1 = NULL;
    FILE *fp2 = NULL;
    fp1 = fopen(cutpath, "wc");
    if (fp1) {
        for (int i = 0; i < (int) m_CutInfo.size(); i++) {
            fprintf(fp1, "%d\n", m_CutInfo[i]);  //切变只用保存发生变化的帧，保存在cutResult.txt中
        }
        fclose(fp1);
    }

    fp2 = fopen(gradientpath, "wc");   //保存在gradientResult.txt中
    if (fp2) {
        for (int i = 0; i < (int) m_GTInfo.size(); i++) {
            //保存了渐变镜头的起始边界和长度
            fprintf(fp2, "%d %d\n", m_GTInfo[i].nStart, m_GTInfo[i].nLength);
        }
        fclose(fp2);
    }

    return 1;
}

vector<int> CShotCut::GetCutInfo(void) {
    return m_CutInfo;
}

vector<GTInfo> CShotCut::GetGTInfo(void) {
    return m_GTInfo;
}

long long CShotCut::GetFrameCount(void) {
    return m_nFrameCount;
}

//int CShotCut::ShotDetection_Old (char* oldcutpath, char* oldgradientpath)
//{
//	//计时参数
//	clock_t start,finish;
//	double totaltime;
//	start = clock();
//	//fprintf(stderr, "镜头分割进行中，请稍候...\n");
//	cout<<"镜头分割进行中，请稍候..."<<endl;
//
//	// 读入原始图像
//	IplImage* pImage = NULL;
//	IplImage* hsv = NULL;
//	IplImage* h_plane = NULL;
//	IplImage* s_plane = NULL;
//	IplImage* v_plane = NULL;
//	//H分量划分为16个等级，S分量划分为16个等级, V分量划分为16个等级
//	int h_bins = HISTBINS, s_bins = HISTBINS, v_bins = HISTBINS;
//	//H分量的变化范围
//	float h_ranges_arr[] = { 0, 180 };
//	float *h_ranges = h_ranges_arr;
//	//S分量的变化范围
//	float s_ranges_arr[] = { 0, 255 };
//	float *s_ranges = s_ranges_arr;
//	//V分量的变化范围
//	float v_ranges_arr[] = { 0, 255 };
//	float *v_ranges = v_ranges_arr;
//	//创建直方图 H维
//	CvHistogram * h_hist = cvCreateHist( 1, &h_bins, CV_HIST_ARRAY, &h_ranges, 1 );
//	//创建直方图 S维
//	CvHistogram * s_hist = cvCreateHist( 1, &s_bins, CV_HIST_ARRAY, &s_ranges, 1 );
//	//创建直方图 V维
//	CvHistogram * v_hist = cvCreateHist( 1, &v_bins, CV_HIST_ARRAY, &v_ranges, 1 );
//	//保存帧直方图数据
//	vector<pHistData> vecHistData;//HistData指针，暂时未使用
//	//上一帧图像数据，用于计算主色率差
//	IplImage *prevImage = NULL;
//	//上一帧直方图数据
//	pHistData	prevHistData = (pHistData)malloc(sizeof(HistData));
//	//当前帧直方图数据
//	pHistData	currHistData = (pHistData)malloc(sizeof(HistData));
//	//渐变起始帧直方图数据，并分配空间
//	pHistData	histGTStart = (pHistData)malloc(sizeof(HistData));
//	//保存滑动窗口的帧间差
//	double	vecSlideWinDiff[SLIDEWINDOWNUMBER]={0.0};
//	//渐变检测过程中保存的帧间差
//	double vecGTDiff[GTWINDOWNUMBER]={0.0};
//	//滑动窗口
//	int	nSlideWindow = 0;
//	//可能渐变标记
//	bool bGT = false;
//	//渐变类型，为0是不包含足球场地，为1是包含足球场地
//	int	nGTType = 0;
//	//渐变检测中的相隔帧差（单调增检测）
////	double dbGTDiff = 0.0;
//	//帧差大于最小阈值的帧数
//	int	nDiffCount = 0;
//	//渐变最大帧差 10
////	int nMaxGTCount = 0;
//	//渐变开始图像帧
//	IplImage *pGTImage = NULL;
//	//渐变开始帧号
//	int nGTStartNum = 0;
//	//渐变结束帧号
////	int	nGTEndNum = 0;
//	//可能切变 如果后一帧的帧差小于最小阈值则认定为切变，否则认为是候选渐变的起始帧
////	bool bCandidacyCut = false;
//	//滑动窗口是否充足
//	bool bSlideEnough = false;
//	//上一帧差
//	double dbLastDiff = 0.0;
//	//是否是新一个镜头的开始 某些切变过渡帧为两帧 dbLastDiff和bIsNewCut处理这种情况
//	bool bIsNewCut = false;
//
//	vector<int>	old_CutInfo;//切变位置
//	vector<GTInfo> old_GTInfo;//渐变信息
//
//	//双阈值
//	double dThresholdHigh1	= 0.0;
//	double dThresholdLow1	= 0.0;
//	double dThresholdHigh2	= 0.0;
//	double dThresholdLow2	= 0.0;
//
//	//文件路径
//	char szPicName[260]={0};
//
//
//
//	for(long long i = 0; i < m_nFrameCount; i++ )
//	{
//		sprintf(szPicName, "%s\\frame%ld.jpg", m_imgPath, i);
//		pImage = cvLoadImage( szPicName, 1 );
//		if(i == 0)
//		{//初始化前一帧和渐变图像
//			m_width = pImage->width;
//			m_height = pImage->height;
//			prevImage = cvCreateImage( cvSize(pImage->width,pImage->height), pImage->depth, pImage->nChannels );
//			pGTImage = cvCreateImage( cvSize(pImage->width, pImage->height), pImage->depth, pImage->nChannels );
//		}
//		//直方图计算
//		if(!hsv)
//		{
//			hsv = cvCreateImage( cvGetSize(pImage), 8, 3 );
//			h_plane = cvCreateImage( cvGetSize(pImage), 8, 1 );
//			s_plane = cvCreateImage( cvGetSize(pImage), 8, 1 );
//			v_plane = cvCreateImage( cvGetSize(pImage), 8, 1 );
//		}
//
//		/** 输入图像转换到HSV颜色空间 */
//		cvCvtColor( pImage, hsv, CV_BGR2HSV );
//		cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
//		/** 计算直方图 H维 */
//		cvCalcHist( &h_plane, h_hist, 0, 0 );
//		/** 计算直方图 S维 */
//		cvCalcHist( &s_plane, s_hist, 0, 0 );
//		/** 计算直方图 V维 */
//		cvCalcHist( &v_plane, v_hist, 0, 0 );
//
//		//获得直方图中的统计次数
//		for(int k = 0; k < h_bins; k++)
//		{
//			currHistData->h[k] = cvQueryHistValue_1D( h_hist, k );
//		}
//		for(int k = 0; k < s_bins; k++)
//		{
//			currHistData->s[k] = cvQueryHistValue_1D( s_hist, k );
//		}
//		for(int k = 0; k < v_bins; k++)
//		{
//			currHistData->v[k] = cvQueryHistValue_1D( v_hist, k );
//		}
//
//		if(i == 0)
//		{//视频第一帧,存储直方图数据
//
//			//复制到上一帧直方图变量
//			CopyHistData( prevHistData, currHistData );
//			////复制第一帧
//			//cvCopyImage(pImage, prevImage);
//			//cvReleaseImage( &pImage );
//		}
//		else
//		{
//			//计算帧间差
//			double curDiff = calFrameHistDiff( prevHistData, currHistData);
//
//
//			//滑动窗口未填充足够，即两个镜头不会离得太近
//			if( nSlideWindow < 16 )
//			{
//				if(nSlideWindow == 0 && !bIsNewCut)
//				{
//					if( curDiff < dbLastDiff/3 )
//					{
//						memmove( &vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) );
//						vecSlideWinDiff[0] = curDiff;
//						nSlideWindow++;
//						CopyHistData( prevHistData, currHistData );
//						bSlideEnough = false;
//					}
//
//					bIsNewCut = true;
//				}
//				else
//				{
//					memmove( &vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) );
//					vecSlideWinDiff[0] = curDiff;
//					nSlideWindow++;
//					CopyHistData( prevHistData, currHistData );
//					bSlideEnough = false;
//				}
//			}
//			else
//			{
//				bSlideEnough = true;
//			}
//
//			dbLastDiff = curDiff;
//
//			//如果不是可能渐变的开始，检测帧间差
//			if( bGT == false && bSlideEnough )
//			{
//				//计算滑动窗口均值和标准差
//				double sum = 0, mean = 0;
//				double variance = 0, stddev = 0;
//
//				if(nSlideWindow == 0)
//				{
//					mean = curDiff;
//					stddev = 0;
//				}
//				else
//				{
//					for(int nsize = 0; nsize < nSlideWindow; nsize++ )
//					{
//						sum += vecSlideWinDiff[nsize];
//					}
//					mean = sum/nSlideWindow;
//					for(int nsize = 0; nsize < nSlideWindow; nsize++ )
//					{
//						variance += (vecSlideWinDiff[nsize] - mean)*(vecSlideWinDiff[nsize] - mean);
//					}
//					stddev = sqrt( variance/nSlideWindow );
//				}
//				//计算高低双阈值
//				dThresholdHigh1 = (mean+THRESHOLD_N*stddev)*LAMBDA_OUTFIELD;
//				dThresholdLow1 = mean*LAMBDA_OUTFIELD;
//				dThresholdHigh2 = (mean+THRESHOLD_N*stddev)*LAMBDA_INFIELD;
//				dThresholdLow2 = mean*LAMBDA_INFIELD;
//
//
//				//当前帧差大于最小低阈值
//				if( curDiff > dThresholdLow2 )
//				{
//					if( calDominatColorRate(pImage, currHistData) <= DOMINATCOLORTHRESHOLD1 )
//					{//不包含足球场地
//						double dDomColorDiff =  calDominatColorDiff(pImage, currHistData, prevImage, prevHistData);
//
//						if( (curDiff > dThresholdHigh1 * 2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh1 * 3) /*|| dDomColorDiff > 0.3*/ )//加入dDomColorDiff > 0.3切变准确率高，渐变检测结果少些
//						{
//							//发生突变
//							nSlideWindow = 0;
//							//记录突变
//							old_CutInfo.push_back( i );
//							bIsNewCut = false;
//						}
//						else if( (curDiff > dThresholdHigh1 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh1 * 2) )
//						{
//							//发生突变
//							nSlideWindow = 0;
//							//记录突变
//							old_CutInfo.push_back( i );
//							bIsNewCut = false;
//						}
//						else if(curDiff <= dThresholdHigh1 && curDiff > dThresholdLow1
//							/*&& dDomColorDiff > DOMINATCOLORTHRESHOLD2*/ )
//						{
//							//可能存在渐变的起始帧
//							bGT = true;
//							nGTType = 0;
//							nDiffCount++;
//							//保存渐变起始帧直方图和起始相隔帧差
//							CopyHistData( histGTStart, prevHistData );
//							cvCopyImage(prevImage, pGTImage);
//							vecGTDiff[GTWINDOWNUMBER-1] = curDiff;
//							nGTStartNum = i;
//						}
//						else
//						{//不是镜头边界，继续增大滑动窗口
//							//保存滑动窗口帧间差
//							memmove( &vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) );
//							vecSlideWinDiff[0] = curDiff;
//							if( nSlideWindow < SLIDEWINDOWNUMBER )
//								nSlideWindow++;
//						}
//					}
//					else
//					{//包含足球场地
//						double dDomColorDiff =  calDominatColorDiff(pImage, currHistData, prevImage, prevHistData);
//						if( (curDiff > dThresholdHigh2 * 2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh2 * 3) /*|| dDomColorDiff > 0.3*/ )//加入dDomColorDiff > 0.3切变准确率高，渐变检测结果少些
//						{
//							//发生突变
//							nSlideWindow = 0;
//							//记录突变
//							old_CutInfo.push_back( i );
//							bIsNewCut = false;
//						}
//						else if( (curDiff > dThresholdHigh2 && dDomColorDiff > DOMINATCOLORTHRESHOLD3) || (curDiff > dThresholdHigh2 * 2) )
//						{
//							//发生突变
//							nSlideWindow = 0;
//							//记录突变
//							old_CutInfo.push_back( i );
//							bIsNewCut = false;
//						}
//						else if(curDiff <= dThresholdHigh2 && curDiff > dThresholdLow2
//							/*&& dDomColorDiff > DOMINATCOLORTHRESHOLD2*/ )
//						{
//							//可能存在渐变的起始帧
//							bGT = true;
//							nGTType = 1;
//							nDiffCount++;
//							//保存渐变起始帧直方图和起始相隔帧差
//							CopyHistData( histGTStart, prevHistData );
//							cvCopyImage(prevImage, pGTImage);
//							vecGTDiff[GTWINDOWNUMBER-1] = curDiff;
//							nGTStartNum = i;
//						}
//						else
//						{//不是镜头边界，继续增大滑动窗口
//							//保存滑动窗口帧间差
//							memmove( &vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) );
//							vecSlideWinDiff[0] = curDiff;
//							if( nSlideWindow < SLIDEWINDOWNUMBER )
//								nSlideWindow++;
//						}
//					}
//				}
//				else
//				{
//					//如果不是渐变检测状态则帧间差存入滑动窗口,如果是渐变在渐变检测中处理
//					//保存滑动窗口帧间差
//					memmove( &vecSlideWinDiff[1], &vecSlideWinDiff[0], sizeof(vecSlideWinDiff) - sizeof(vecSlideWinDiff[0]) );
//					vecSlideWinDiff[0] = curDiff;
//					if( nSlideWindow < SLIDEWINDOWNUMBER )
//						nSlideWindow++;
//				}
//
//			}
//			else if( bSlideEnough )
//			{//正在渐变检测过程中
//				//相隔帧超过阈值
//				double dBeforeDomColorDiff =  calDominatColorDiff(pImage, currHistData, pGTImage, histGTStart);
//				if(curDiff > (nGTType==0 ? dThresholdLow1 : dThresholdLow2) )
//				{
//					if( calFrameHistDiff( histGTStart, currHistData ) > (nGTType==0 ? dThresholdHigh1 : dThresholdHigh2) &&
//						dBeforeDomColorDiff > DOMINATCOLORTHRESHOLD2 )
//					{
//						//渐变终止帧，检测出渐变
//						bGT = false;
//						GTInfo tmpInfo;
//						tmpInfo.nStart = nGTStartNum;
//						tmpInfo.nLength = i - nGTStartNum + 1;
//						if(tmpInfo.nLength > 2)
//						{
//							old_GTInfo.push_back( tmpInfo );
//							bIsNewCut = false;
//
//							nSlideWindow = 0;
//						}
//					}
//				}
//				else
//				{
//					bGT = false;
//				}
//
//			}
//
//			//释放上一个直方图数据，重新赋值
//			CopyHistData( prevHistData, currHistData );
//		}
//		cvCopyImage(pImage, prevImage);
//		cvReleaseImage( &pImage );
//	}
//
//	//最后一帧算是镜头结尾
//	old_CutInfo.push_back( m_nFrameCount-1 );
//
//	//cvReleaseImage( &pImage );
//	cvReleaseImage( &prevImage );
//	cvReleaseImage( &pGTImage );
//	cvReleaseImage( &hsv );
//	cvReleaseImage( &h_plane );
//	cvReleaseImage( &s_plane );
//	cvReleaseImage( &v_plane );
//	free(prevHistData);
//	free(currHistData);
//	free(histGTStart);
//	cvReleaseHist( &h_hist );
//	cvReleaseHist( &s_hist );
//	cvReleaseHist( &v_hist );
//
//	FILE* fp1 = NULL;
//	FILE* fp2 = NULL;
//	fp1 = fopen( oldcutpath, "wc" );
//	if(fp1)
//	{
//		for(int i=0; i < old_CutInfo.size(); i++)
//		{
//			fprintf( fp1, "%d\n", old_CutInfo[i] );
//		}
//		fclose(fp1);
//	}
//
//	fp2 = fopen( oldgradientpath, "wc" );
//	if(fp2)
//	{
//		for(int i=0; i < old_GTInfo.size(); i++)
//		{
//			fprintf( fp2, "%d %d\n", old_GTInfo[i].nStart, old_GTInfo[i].nLength );
//		}
//		fclose(fp2);
//	}
//
//	finish = clock();
//	totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
//	//fprintf(stderr, "镜头分割完成！共耗时%.02lf秒！\n", totaltime );
//	cout<<"镜头分割完成！共耗时"<<totaltime<<"秒！"<<endl;
//	return 1;
//}
