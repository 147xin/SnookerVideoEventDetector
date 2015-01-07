//#include "StdAfx.h"
#include "ReplayDetector.h"

//按值降序排列
bool greater_value(const SumPoint &c1, const SumPoint &c2) {
    return c1.value > c2.value;
}

//按值升序排列
bool less_value(const SumPoint &c1, const SumPoint &c2) {
    return c1.value < c2.value;
}

//按坐标降序排列
bool greater_coordinate(const SumPoint &c1, const SumPoint &c2) {
    bool res = false;
    if (c1.x > c2.x) {
        res = true;
    }
    else if (c1.x == c2.x) {
        if (c1.y > c2.y) {
            res = true;
        }
    }
    return res;
}

//按坐标升序排列
bool less_coordinate(const SumPoint &c1, const SumPoint &c2) {
    bool res = false;
    if (c1.x < c2.x) {
        res = true;
    }
    else if (c1.x == c2.x) {
        if (c1.y < c2.y) {
            res = true;
        }
    }
    return res;
}

CReplayDetector::CReplayDetector(void) {
    m_RframePos = NULL;
    m_pMaskImage = NULL;
    m_logoFlag = NULL;
    m_scFrame = NULL;
    m_logoCount = 0;
    m_nMaxSeqPos = -1;
    m_nKframePos = -1;
    m_nVideoFrames = 0;
    m_nLpixelsCount = 0;
}

CReplayDetector::~CReplayDetector(void) {
    if (m_RframePos) {
        delete[] m_RframePos;
        m_RframePos = NULL;
    }
    if (m_pMaskImage) {
        cvReleaseImage(&m_pMaskImage);
        m_pMaskImage = NULL;
    }
    if (m_logoFlag) {
        delete[] m_logoFlag;
        m_logoFlag = NULL;
    }
    for (int i = 0; i < (int) m_SeqFlow.size(); i++) {
        PBlockFlow *pFrameFlow = NULL;
        pFrameFlow = m_SeqFlow[i];
        for (int j = 0; j < m_vecGTInfo[i].nLength - 1; j++) {
            delete[] pFrameFlow[j];
        }
        delete pFrameFlow;
        pFrameFlow = NULL;
    }

    if (m_scFrame) {
        double **dbSC = NULL;
        int m_vecGTInfo_len = m_vecGTInfo.size();
        for (int i = 0; i < m_vecGTInfo_len; i++) {
            for (int j = i + 1; j < m_vecGTInfo_len; j++) {
                dbSC = m_scFrame[i][j];
                for (int m = 0; m < m_vecGTInfo[i].nLength; m++) {
                    delete[] dbSC[m];
                }
                delete[] dbSC;
            }
            delete[] m_scFrame[i];
        }
        delete[] m_scFrame;
    }
}

int CReplayDetector::SetPath(char *gradientpath, char *cutpath, char *imgpath) {
    if (gradientpath == NULL || cutpath == NULL || imgpath == NULL) {
        return 0;
    }

    strcpy(m_gradientPath, gradientpath);
    strcpy(m_cutPath, cutpath);
    strcpy(m_imgPath, imgpath);

    start = clock();
    fprintf(stdout, "Replay Shot Detecting...\n");
    return 1;
}

int CReplayDetector::SetVideoFrames(int framecount) {
    if (framecount > 0) {
        m_nVideoFrames = framecount;
        return 1;
    }
    else
        return 0;
}

//读取切变信息
int CReplayDetector::ReadFile(void) {
    FILE *fp = NULL;

    //读取切变
    int cutpos;
    fp = fopen(m_cutPath, "r");
    if (fp == NULL) {
        printf("File is not exist");
        return 0;
    }
    while (fscanf(fp, "%d", &cutpos) != EOF) {
        m_vecCutInfo.push_back(cutpos);
    }
    fclose(fp);

    return 1;
}

//求3个数中最大值
double CReplayDetector::max3(double a, double b, double c) {
    if (a >= b && a >= c)
        return a;
    if (b >= a && b >= c)
        return b;
    else
        return c;
}

/**
*  计算所有渐变序列的光流
*
*  @return 1表示成功完成
*/
int CReplayDetector::CalcOpticalFlow(void) {
    IplImage *pShowImg[MAXFRAMECOUNT]; //为了绘图 测试用???代码里没有用到这个
    IplImage *pColorImg = NULL;

    IplImage *pSequenes[MAXFRAMECOUNT];
    IplImage *pTmpImage = NULL;
    char strImgPath[260] = {0};
    int nBlocks = (IMAGEWIDTH / nWinSize) * (IMAGEHEIGHT / nWinSize);
    BlockFlow *dbBlockFlow = new BlockFlow[nBlocks]; // xx, yy, alpha_angle
    memset(dbBlockFlow, 0, nBlocks * sizeof(BlockFlow));

    for (int i = 0; i < (int) m_vecGTInfo.size(); i++) {
        //读取图像序列
        for (int j = 0; j < m_vecGTInfo[i].nLength; j++) {
            if ((m_vecGTInfo[i].nStart + j) >= m_nVideoFrames) {
                break;
            }

            sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath,
                    m_vecGTInfo[i].nStart + j);
            //读取原图像(灰度)
            pTmpImage = cvLoadImage(strImgPath, 0);
            //创建新的指定大小图像
            pSequenes[j] = cvCreateImage(cvSize(IMAGEWIDTH, IMAGEHEIGHT),
                    pTmpImage->depth, pTmpImage->nChannels);
            //缩放原图像(灰度)
            cvResize(pTmpImage, pSequenes[j], 1);
            //释放原图像
            cvReleaseImage(&pTmpImage);

            //下面这些是测试时用的吗???
            //读取原图像(彩色)
            pColorImg = cvLoadImage(strImgPath, 1);
            //创建新的指定大小图像
            pShowImg[j] = cvCreateImage(cvSize(IMAGEWIDTH, IMAGEHEIGHT),
                    pColorImg->depth, pColorImg->nChannels);
            //缩放原图像(彩色)
            cvResize(pColorImg, pShowImg[j], 1);
            cvReleaseImage(&pColorImg);
        }

        //计算图像序列光流
        PBlockFlow *pFrameFlow = new PBlockFlow[m_vecGTInfo[i].nLength - 1]; //存放当前渐变序列的光流, 长度为渐变序列长度减1
        for (int j = 1; j < m_vecGTInfo[i].nLength; j++) {
            ImgOpticalFlow(pSequenes[j - 1], pSequenes[j], nWinSize, dbBlockFlow); //计算相邻两帧之间的光流

            //hix test 将光流值显示出来以便观察结果
            int xxx = IMAGEWIDTH / nWinSize, yyy = IMAGEHEIGHT / nWinSize;
            for (int iii = 0; iii < yyy; ++iii) {
                for (int jjj = 0; jjj < xxx; ++jjj) {
                    cout << abs(dbBlockFlow[iii * xxx + jjj].xvalue) + abs(dbBlockFlow[iii * xxx + jjj].yvalue) << "_"
                            << dbBlockFlow[iii * xxx + jjj].direction << "   ";
                }
                cout << endl;
            }
            cout << endl;
            //test end

            pFrameFlow[j - 1] = new BlockFlow[MAXBLOCKCOUNT]; //这个MAXBLOCKCOUNT（8）比nBlocks（48）要小，意味着只存储部分blocks的光流数据
            memset(pFrameFlow[j - 1], 0, MAXBLOCKCOUNT * sizeof(BlockFlow)); //清零

            //将dbBlockFlow（相邻两帧全部块的光流）中每个块的值与pFrameFlow[j-1]（当前渐变序列的光流）中的每个值进行比较
            //pFrameFlow[j-1]中的最终结果是前MAXBLOCKCOUNT大的光流块，并且按大小升序排列（这里实际上用的是插入排序）
            for (int m = 0; m < nBlocks; m++) {
                for (int n = 0; n < MAXBLOCKCOUNT; n++) {
                    if (abs(dbBlockFlow[m].xvalue) + abs(dbBlockFlow[m].yvalue)
                            > abs(pFrameFlow[j - 1][n].xvalue) + abs(pFrameFlow[j - 1][n].yvalue)) {
                        //n大于0
                        if (n > 0) {
                            pFrameFlow[j - 1][n - 1] = pFrameFlow[j - 1][n];
                            pFrameFlow[j - 1][n] = dbBlockFlow[m];
                        }
                            //如果n为0，则用dbBlockFlow[m]替换pFrameFlow[j - 1][0]
                        else {
                            pFrameFlow[j - 1][0] = dbBlockFlow[m];
                        }
                    }
                }
            }

        } // for

        //保存当前渐变序列的光流值
        m_SeqFlow.push_back(pFrameFlow);

        //释放图像序列
        for (int j = 0; j < m_vecGTInfo[i].nLength; j++) {
            cvReleaseImage(&pSequenes[j]);
        }
        for (int j = 0; j < m_vecGTInfo[i].nLength; j++) {
            cvReleaseImage(&pShowImg[j]);
        }
    }

    delete[] dbBlockFlow;
    return 1;
}

int CReplayDetector::CalcDirection(double xx, double yy) {
    double alpha_angle;
    if (xx < ZERO && xx > -ZERO) {
        if (yy < ZERO && yy > -ZERO)
            return 0;
        else
            alpha_angle = pi / 2;
    }
    else {
        alpha_angle = abs(atan(yy / xx));
    }

    if (xx < 0 && yy > 0)
        alpha_angle = pi - alpha_angle;
    if (xx < 0 && yy < 0)
        alpha_angle = pi + alpha_angle;
    if (xx > 0 && yy < 0)
        alpha_angle = 2 * pi - alpha_angle;

    int direction = (int) (alpha_angle / (2 * pi / 16) + 1);
    if (direction > 16)
        direction = 16;
    return direction;
}

int CReplayDetector::ImgOpticalFlow(IplImage *prev_grey, IplImage *grey,
        int nWinSize, BlockFlow *dbBlockFlow) {
    // IplImage *image = cvCloneImage(prev_grey);
    IplImage *velx = cvCreateImage(cvSize(grey->width, grey->height), IPL_DEPTH_32F, 1);
    IplImage *vely = cvCreateImage(cvSize(grey->width, grey->height), IPL_DEPTH_32F, 1);
    velx->origin = vely->origin = grey->origin;

    CvSize winSize = cvSize(5, 5);
    cvCalcOpticalFlowLK(prev_grey, grey, winSize, velx, vely);

    const int winsize = nWinSize; //计算光流的窗口大小
    int id = 0;
    BlockFlow *pTmpFlow = dbBlockFlow;
    for (int y = 0; y < velx->height; y += winsize)
        for (int x = 0; x < velx->width; x += winsize) {
            if (x >= 0 && y >= 0 && x <= velx->width - winsize && y <= velx->height - winsize) {
                cvSetImageROI(velx, cvRect(x, y, winsize, winsize));
                CvScalar total_x = cvSum(velx);
                double xx = total_x.val[0];
                cvResetImageROI(velx);
                cvSetImageROI(vely, cvRect(x, y, winsize, winsize));
                CvScalar total_y = cvSum(vely);
                double yy = total_y.val[0];
                cvResetImageROI(vely);

                //统计方向
                int nDirection[17] = {0};
                CvScalar tmpX, tmpY;
                double dbx, dby;
                for (int i = 0; i < winsize; i++) {
                    for (int j = 0; j < winsize; j++) {
                        tmpX = cvGet2D(velx, y + j, x + i);
                        tmpY = cvGet2D(vely, y + j, x + i);
                        dbx = tmpX.val[0];
                        dby = tmpY.val[0];
                        int direction = CalcDirection(dbx, dby);
                        nDirection[direction]++;
                    }
                }
                int maxDirectValue = 0;
                int maxDirectPos = 0;
                for (int i = 0; i < 17; i++) {
                    if (maxDirectValue < nDirection[i]) {
                        maxDirectValue = nDirection[i];
                        maxDirectPos = i;
                    }
                }

                pTmpFlow->xvalue = xx;
                pTmpFlow->yvalue = yy;
                pTmpFlow->direction = maxDirectPos;
                pTmpFlow->id = id;
                id++;
                pTmpFlow++;
            }
        }

    cvReleaseImage(&velx);
    cvReleaseImage(&vely);
    // cvReleaseImage(&image);

    return 1;
}

//不同序列的光流相似性
double CReplayDetector::SC_frames(BlockFlow *frame1, BlockFlow *frame2) {
    //比较top 10%方向  可是这里并没有比较top10%的方向啊！！！这里是比较了所有的块啊！！！
    int nMatchBlocks = 0;
    for (int i = 0; i < MAXBLOCKCOUNT; i++) {
        for (int j = 0; j < MAXBLOCKCOUNT; j++) {
            if (frame1[i].id == frame2[j].id && frame1[i].direction == frame2[j].direction && abs(frame1[i].xvalue) > 0 && abs(frame1[i].yvalue) > 0 && abs(frame2[j].xvalue) > 0 && abs(frame2[j].yvalue) > 0)
                nMatchBlocks++;
        }
    }

    return (double) nMatchBlocks / MAXBLOCKCOUNT;
}

/**
*  序列比对, 计算匹配值
*
*  @param firstPos  第一个渐变序列在所有渐变序列中的序号
*  @param secondPos 第二个渐变序列在所有渐变序列中的序号
*  @param nSeqLen1  第一个渐变序列的长度
*  @param nSeqLen2  第二个渐变序列的长度
*
*  @return 匹配值
*/
double CReplayDetector::Alignment(int firstPos, int secondPos, int nSeqLen1,
        int nSeqLen2) {
    double dbScore[MAXFRAMECOUNT][MAXFRAMECOUNT] = {0};
    double a = 0, b = 0, c = 0;
    double tmpSC = 0.0;
    int pairs = 0;
    double AlignRes = 0.0;
    bool bSimilar = false;

    //待释放
    double **dbSC = new double *[nSeqLen1];
    for (int i = 0; i < nSeqLen1; i++) {
        dbSC[i] = new double[nSeqLen2];
        for (int j = 0; j < nSeqLen2; j++) {
            dbSC[i][j] = 0;
        }
    }

    for (int i = 1; i < nSeqLen1; i++) {
        for (int j = 1; j < nSeqLen2; j++) {
            tmpSC = SC_frames(m_SeqFlow[firstPos][i - 1], m_SeqFlow[secondPos][j - 1]);
            a = dbScore[i][j - 1];
            b = dbScore[i - 1][j];
            c = dbScore[i - 1][j - 1] + tmpSC;
            dbScore[i][j] = max3(a, b, c);

            if (tmpSC > 0.3)
                pairs++;

            // add for yingchao 12.9 多写几句注释会死啊！！！
            if (tmpSC >= 0.5)
                bSimilar = true;

            dbSC[i - 1][j - 1] = tmpSC;
        }
    }

    //保存序列帧间相似度，最后释放
    m_scFrame[secondPos][firstPos] = m_scFrame[firstPos][secondPos] = dbSC;

    //匹配对至少大于8
    if (bSimilar && pairs >= 3 /*8*/ && pairs <= 15 && dbScore[nSeqLen1 - 1][nSeqLen2 - 1] > 2.0) //2.0怎么来的?
        AlignRes = dbScore[nSeqLen1 - 1][nSeqLen2 - 1] /*/pairs*/; //为什么把pairs注释掉了???

    return AlignRes;
}

/**
*  获取Logo template
*
*  @return 1表示成功完成
*/
int CReplayDetector::GetLogoTemplate(void) {
    double dbMaxScore = 0, dbMinScore = 1000, dbAvgScore = 0;

    //总的渐变个数
    int m_vecGTInfoLen = (int) m_vecGTInfo.size();

    //指示各个渐变是否是Logo
    m_logoFlag = new int[m_vecGTInfoLen];
    memset(m_logoFlag, 0, m_vecGTInfoLen * sizeof(int));

    //一行的最大值???
    int *rowMax = new int[m_vecGTInfoLen];
    memset(rowMax, 0, m_vecGTInfoLen * sizeof(int));

    double dbMaxSum = 0;
    int nMaxPos = -1;

    double **dbAlign = new double *[m_vecGTInfoLen];
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        dbAlign[i] = new double[m_vecGTInfoLen];
        for (int j = 0; j < m_vecGTInfoLen; j++)
            dbAlign[i][j] = 0;
    }
    //计算序列比对
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        for (int j = i + 1; j < m_vecGTInfoLen; j++) {
            //计算序列匹配值
            dbAlign[j][i] = dbAlign[i][j] = Alignment(i, j, m_vecGTInfo[i].nLength, m_vecGTInfo[j].nLength);

            //记下最大值与最小值
            if (dbAlign[j][i] > dbMaxScore) {
                dbMaxScore = dbAlign[j][i];
            }
            else if (dbAlign[j][i] < dbMinScore && dbMinScore > ZERO) {
                dbMinScore = dbAlign[j][i];
            }
        }
    }
    //大于平均值，认为是logo序列
    m_dbAvgScore = dbAvgScore = max(dbMaxScore / 4, 2.4); //这个2.4是怎么来的???
    int maxRow = 0;
    //找出大于平均值最多次数的那一行
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        for (int j = 0; j < m_vecGTInfoLen; j++) {
            if (dbAlign[i][j] > dbAvgScore) {
                rowMax[i]++;
                // m_logoFlag[i] = m_logoFlag[j] = 1;
            }
        }
        if (maxRow < rowMax[i])
            maxRow = rowMax[i];
    }
    //大于maxRow/2被认为是Logo序列
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        if (rowMax[i] > maxRow / 2)
            m_logoFlag[i] = 1;
    }

    //hix test 输出m_logoFlag
    cout << "m_logoFlag: ";
    for (int iii = 0; iii < m_vecGTInfoLen; ++iii) {
        cout << m_logoFlag[iii] << " ";
    }
    cout << endl;
    //test end

    //求logo template，它是与其他序列匹配度之和最大的那个序列
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        if (m_logoFlag[i]) {
            double tmpSum = 0;
            for (int j = 0; j < m_vecGTInfoLen; j++) {
                if (m_logoFlag[j])
                    tmpSum += dbAlign[i][j];
            }
            if (dbMaxSum < tmpSum) {
                dbMaxSum = tmpSum;
                nMaxPos = i;
            }
            m_logoCount++; // logo序列的个数
        }
    }
    //Logo-template
    m_nMaxSeqPos = nMaxPos;

    //释放空间
    for (int i = 0; i < m_vecGTInfoLen; i++) {
        delete[] dbAlign[i];
    }
    delete[] dbAlign;

    //判断是否找到logo template
    if (m_nMaxSeqPos < 0) {
        printf("Not find St");
        return 0;
    }
    return 1;
}

//获取L-pixels
int CReplayDetector::GetLpixels(void) {
    // int	nFramePos = -1;
    SCSEQUENES *seqSC = NULL;
    SCSEQUENES seqSCseq = NULL;
    double dbMaxFrameSC = 0.0;
    int nMaxFramePos = 0;
    int nPixelCount = 0;

    //判断是否找到logo template
    if (m_nMaxSeqPos < 0)
        return 0;

    seqSC = m_scFrame[m_nMaxSeqPos]; //Logo Template序列与其他序列帧之间的相似度
    //获得K-frame
    for (int i = 0; i < (int) m_vecGTInfo.size(); i++) {
        if (m_logoFlag[i] && i != m_nMaxSeqPos) { //如果序列i是logo序列, 并且不是logo template序列
            //序列与序列的帧间相似度
            seqSCseq = seqSC[i]; //Logo Template序列与序列i之间的帧间相似度

            for (int j = 0; j < m_vecGTInfo[m_nMaxSeqPos].nLength; j++) { //j是logoTemplate序列中的每一帧
                for (int k = 0; k < m_vecGTInfo[i].nLength; k++) { //k是序列i的每一帧
                    //序列比对时申请的空间，double m_scFrame[s1][s2][s1_len][s2_len]
                    //且s1<s2
                    if (i > m_nMaxSeqPos && dbMaxFrameSC < seqSCseq[j][k]) {
                        dbMaxFrameSC = seqSCseq[j][k];
                        nMaxFramePos = j + 1;
                    }
                    else if (i < m_nMaxSeqPos && dbMaxFrameSC < seqSCseq[k][j]) {
                        dbMaxFrameSC = seqSCseq[k][j];
                        nMaxFramePos = j + 1;
                    }
                }
            }
        }
    }
    // Kframe在最大序列比对渐变(Logo Template)中的位置
    m_nKframePos = nMaxFramePos;

    //获得R-frame 问题可能出在这里！！！hix
    //用以存放R-frame的检测结果
    m_RframePos = new int[m_vecGTInfo.size()];
    memset(m_RframePos, 0, m_vecGTInfo.size() * sizeof(int));

    for (int i = 0; i < (int) m_vecGTInfo.size(); i++) { //对于每个渐变i
        double curSC = 0;
        double tmpSC = 0;
        int tmpPos = 0;
        if (m_logoFlag[i] && i != m_nMaxSeqPos) { //如果渐变i是logo序列, 并且不是logo template
            //序列与序列的帧间相似度
            seqSCseq = seqSC[i]; //Logo Template序列与序列i之间的帧间相似度
            for (int k = 0; k < m_vecGTInfo[i].nLength; k++) { //k是序列i的每一帧
                if (i < m_nMaxSeqPos)
                    curSC = seqSCseq[k][nMaxFramePos];
                else
                    curSC = seqSCseq[nMaxFramePos][k];
                if (tmpSC < curSC) {
                    tmpSC = curSC;
                    tmpPos = k;
                }
            }
            // m_RframePos[i] = tmpPos;
            if (tmpSC >= /*0.5*/ 0.375) { //这里本来是0.5
                m_logoFlag[i] = 1;
                m_RframePos[i] = tmpPos;
            }
            else {
                m_logoFlag[i] = 0;
            }
        }
    }
    //test 输出RframePos的值hix
    cout << "R-frames: ";
    for (int iii = 0; iii < m_vecGTInfo.size(); ++iii) {
        cout << *(m_RframePos + iii) << " ";
    }
    cout << endl;


    //读取K-frame和R-frame图像
    vector<IplImage *> vecFrames; //存放K-frame与R-frame
    vector<IplImage *> vecNframes; //存放N-frame
    IplImage *pTmpImage = NULL;
    IplImage *pDiffImage = NULL;
    int vecsize = 0;
    char strImgPath[260] = {0};
    // K-frame
    if (m_vecGTInfo[m_nMaxSeqPos].nStart + nMaxFramePos < m_nVideoFrames) { //如果K-frame的位置小于视频最大帧数
        sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath,
                m_vecGTInfo[m_nMaxSeqPos].nStart + nMaxFramePos);
        pTmpImage = cvLoadImage(strImgPath, 1);
        vecFrames.push_back(pTmpImage); //将K-frame加入vecFrames
    }

    // N-frame
    int framenum = m_vecGTInfo[m_nMaxSeqPos].nStart - 30;
    framenum = (framenum > 0) ? framenum : 0; //确保framenum>=0
    sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath, framenum);
    pTmpImage = cvLoadImage(strImgPath, 1);
    vecNframes.push_back(pTmpImage); //加入vecNframes

    //将其他R-frames与N-frames加入对应的vector
    for (int i = 0; i < (int) m_vecGTInfo.size(); i++) { //i为每一个渐变序列号
        if (m_logoFlag[i] && i != m_nMaxSeqPos) { //如果是logo且不是logo-template
            // R-frame
            if (m_vecGTInfo[i].nStart + m_RframePos[i] < m_nVideoFrames) { //渐变i的R-frame
                sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath,
                        m_vecGTInfo[i].nStart + m_RframePos[i]);
                pTmpImage = cvLoadImage(strImgPath, 1);
                vecFrames.push_back(pTmpImage);
            }
            // N-frame
            framenum = m_vecGTInfo[i].nStart - 30;  //渐变i的N-frame
            framenum = (framenum > 0) ? framenum : 0;

            sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath, framenum);
            pTmpImage = cvLoadImage(strImgPath, 1);
            vecNframes.push_back(pTmpImage);
        }
    }

    //计算K-frame R-frame像素差之和
    IplImage *pSumImage = cvCreateImage(cvGetSize(pTmpImage), IPL_DEPTH_32F, pTmpImage->nChannels);
    IplImage *pScaleImage = cvCreateImage(cvGetSize(pTmpImage), IPL_DEPTH_32F, pTmpImage->nChannels);
    cvZero(pScaleImage);
    cvZero(pSumImage);
    pDiffImage = cvCloneImage(pTmpImage);

    vecsize = (int) vecFrames.size();
    nPixelCount = pSumImage->width * pSumImage->height;

    // char FrameSavePath[256] = {0};

    for (int i = 0; i < vecsize; i++) {
        for (int j = i + 1; j < vecsize; j++) {
            cvAbsDiff(vecFrames[i], vecFrames[j], pDiffImage);
            cvScale(pDiffImage, pScaleImage);
            cvAdd(pScaleImage, pSumImage, pSumImage);
        }
    }

    //像素排序 升序 取前30%~50%
    int nC1Count = (int) (nPixelCount * 0.4);
    SumPoint tmpPoint;
    vector<SumPoint> vecSumPointC1(pSumImage->width * pSumImage->height);
    //遍历pSumImage
    for (int height = 0; height < pSumImage->height; height++) {
        const float *ptr = (const float *) (pSumImage->imageData + height * pSumImage->widthStep);
        for (int width = 0; width < pSumImage->width; width++) {
            tmpPoint.value = ptr[3 * width] + ptr[3 * width + 1] + ptr[3 * width + 2]; //3为RGB3个通道
            tmpPoint.x = width;
            tmpPoint.y = height;
            vecSumPointC1[height * pSumImage->width + width] = tmpPoint;
        }
    }

    // vector升序排序, 前nC1Count的元素是升序排列的
    partial_sort(vecSumPointC1.begin(), vecSumPointC1.begin() + nC1Count,
            vecSumPointC1.end(), less_value);

    //计算K-frame和R-frame与N-frame的像素差之和
    cvZero(pScaleImage);
    cvZero(pSumImage);
    cvZero(pDiffImage);

    for (int i = 0; i < vecsize; i++) {
        for (int j = 0; j < (int) vecNframes.size(); j++) {
            cvAbsDiff(vecFrames[i], vecNframes[j], pDiffImage);
            cvScale(pDiffImage, pScaleImage);
            cvAdd(pScaleImage, pSumImage, pSumImage);
        }
    }

    //像素排序 降序 取前30%~50%
    int nC2Count = (int) (nPixelCount * 0.4);
    SumPoint tmpNframePoint;
    vector<SumPoint> vecSumPointC2(pSumImage->width * pSumImage->height);
    for (int height = 0; height < pSumImage->height; height++) {
        const float *ptr = (const float *) (pSumImage->imageData + height * pSumImage->widthStep);
        for (int width = 0; width < pSumImage->width; width++) {
            tmpNframePoint.value = ptr[3 * width] + ptr[3 * width + 1] + ptr[3 * width + 2];
            tmpNframePoint.x = width;
            tmpNframePoint.y = height;
            vecSumPointC2[height * pSumImage->width + width] = tmpNframePoint;
        }
    }

    // vector降序排序
    partial_sort(vecSumPointC2.begin(), vecSumPointC2.begin() + nC2Count,
            vecSumPointC2.end(), greater_value);

    //两个向量按照坐标值升序排序并求交集
    sort(vecSumPointC1.begin(), vecSumPointC1.begin() + nC1Count,
            less_coordinate);
    sort(vecSumPointC2.begin(), vecSumPointC2.begin() + nC2Count,
            less_coordinate);
    vector<SumPoint> vecMixPoint(nC1Count);
    m_nLpixelsCount = set_intersection(vecSumPointC1.begin(), vecSumPointC1.begin() + nC1Count,
            vecSumPointC2.begin(), vecSumPointC2.begin() + nC2Count,
            vecMixPoint.begin(), less_coordinate) - vecMixPoint.begin();

    //释放空间
    for (int i = 0; i < (int) vecFrames.size(); i++) {
        cvReleaseImage(&vecFrames[i]);
    }
    for (int i = 0; i < (int) vecNframes.size(); i++) {
        cvReleaseImage(&vecNframes[i]);
    }
    cvReleaseImage(&pDiffImage);
    cvReleaseImage(&pSumImage);
    cvReleaseImage(&pScaleImage);

    if (m_nLpixelsCount < nPixelCount * 0.06) { //这里原来是0.1
        printf("Not find L-pixels");
        return 0;
    }

    //构造L-pixels的掩码图像
    m_pMaskImage = cvCreateImage(cvGetSize(pTmpImage), IPL_DEPTH_8U, 1);
    cvZero(m_pMaskImage);
    vector<SumPoint>::const_iterator it = vecMixPoint.begin();
    for (; it != vecMixPoint.end(); ++it) {
        *(m_pMaskImage->imageData + (*it).y * m_pMaskImage->widthStep + (*it).x) = 255;
    }
//    cvShowImage("L-pixels Mask", m_pMaskImage);
//    cvWaitKey();

    return 1;
}

int CReplayDetector::LogoDetection(void) {
    IplImage *pTmpImage = NULL;
    IplImage *pKframe = NULL;
    IplImage *pDiffImage = NULL;
    char strImgPath[256] = {0};

    //读取K-Frame
    if (m_vecGTInfo[m_nMaxSeqPos].nStart + m_nKframePos < m_nVideoFrames) {
        sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath,
                m_vecGTInfo[m_nMaxSeqPos].nStart + m_nKframePos);
        pKframe = cvLoadImage(strImgPath, 1);
        //创建pDiffImage;
        pDiffImage = cvCreateImage(cvGetSize(pKframe), pKframe->depth, pKframe->nChannels);
        cvZero(pDiffImage);
    }

    int nMaxReplayCount = 300;
    int *pProbLogoFrames = new int[nMaxReplayCount]; //帧编号
    double *pProbFrameSub = new double[nMaxReplayCount]; //帧编号对应的平均帧差
    bool *pRframeFlag = new bool[nMaxReplayCount]; //是否为R-frame
    memset(pProbLogoFrames, 0, nMaxReplayCount * sizeof(int));
    memset(pProbFrameSub, 0, nMaxReplayCount * sizeof(double));
    for (int i = 0; i < nMaxReplayCount; i++) {
        pProbFrameSub[i] = 100000000;
    }
    //判断L-pixels像素个数是否合理
    if (m_nLpixelsCount <= 0)
        return 0;

    //获取L-pixels差比较小的帧
    int nTmpReplay = 0;

    //读取每一个视频帧, 与K-frame求帧差, 若帧差小于一个阈值, 则将该视频帧与帧差值保存下来并按帧差大小升序排列
    for (int i = 0; i < m_nVideoFrames; i++) {
        sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath, i);
        pTmpImage = cvLoadImage(strImgPath, 1);
        //与K-frame作差
        cvSub(pKframe, pTmpImage, pDiffImage, m_pMaskImage);
        cvAbs(pDiffImage, pDiffImage);
        //计算各个通道之和
        CvScalar sum = cvSum(pDiffImage);
        //每个像素在各个通道的均值
        double dbSum = (sum.val[0] + sum.val[1] + sum.val[2]) / m_nLpixelsCount / pKframe->nChannels;
        cvReleaseImage(&pTmpImage);
        pTmpImage = NULL;

        //平均帧差小于阈值，才可能是logo图像
        if (dbSum < 15) { //这个阈值是如何确定的?
            nTmpReplay++;
            //将dbSum和i插入到pProbFrameSub和pProbLogoFrames中, 按升序排列, 这里用了插入排序
            for (int n = nMaxReplayCount - 1; n >= 0; n--) {
                if (dbSum < pProbFrameSub[n]) {
                    if (n != nMaxReplayCount - 1) { //n不是尾元素
                        pProbFrameSub[n + 1] = pProbFrameSub[n];
                        pProbFrameSub[n] = dbSum;
                        pProbLogoFrames[n + 1] = pProbLogoFrames[n];
                        pProbLogoFrames[n] = i;
                    }
                    else {
                        pProbFrameSub[n] = dbSum;
                        pProbLogoFrames[n] = i;
                    }
                }
            }
        } // if
    } // for
    if (nTmpReplay > nMaxReplayCount)
        nTmpReplay = nMaxReplayCount;

    //初始化pRframeFlag数组, 指示每个帧是不是R-frame, 初始时都为true, 后面根据相邻的间隔过滤掉一些
    for (int i = 0; i < nTmpReplay; i++) {
        pRframeFlag[i] = true;
    }
    vector<int> candidateLogo; //候选logo
    //帧差过滤，如果相隔帧小于20，则认为是同属于一个渐变，后面的帧标记为非R-frame
    for (int i = 0; i < nTmpReplay; i++) {
        if (pRframeFlag[i] == true) {
            candidateLogo.push_back(pProbLogoFrames[i]);
        }
        else {
            continue;
        }
        for (int j = i + 1; j < nTmpReplay; j++) {
            if (abs(pProbLogoFrames[i] - pProbLogoFrames[j]) < 20) {
                pRframeFlag[j] = false;
            }
        }
    }

    //序列比对
    int candidateLogoLen = (int) candidateLogo.size();
    for (int i = 0; i < candidateLogoLen; i++) {
        double AlignValue = 0;
        GTInfo tmpInfo;
        tmpInfo.nStart = candidateLogo[i] - m_nKframePos;
        tmpInfo.nLength = m_vecGTInfo[m_nMaxSeqPos].nLength;

        // cout << "I:"<<i<<"  Start:" << tmpInfo.nStart  << " Length:" <<
        // tmpInfo.nLength << endl;
        if (tmpInfo.nStart < 1 || tmpInfo.nLength < 1)
            continue;

        AlignValue = VerifyLogoSeq(tmpInfo);
        //序列比对值大于阈值
        if (AlignValue > m_dbAvgScore)
            m_confirmLogo.push_back(tmpInfo.nStart);
    }

    //m_confirmLogo中的帧序号并没有按从小到大进行排序, 这会导致后面进行logo配对时出错
    //现对其进行排序 added by 黄易欣 on 2014-12-12
    sort(m_confirmLogo.begin(), m_confirmLogo.end());

    delete[] pProbLogoFrames;
    delete[] pProbFrameSub;
    delete[] pRframeFlag;

    cvReleaseImage(&pKframe);
    cvReleaseImage(&pDiffImage);
    return 1;
}

/**
*  验证渐变序列是否确实是logo序列
*
*  @param info 渐变序列
*
*  @return 匹配度值
*/
double CReplayDetector::VerifyLogoSeq(GTInfo info) {
    IplImage *pSequenes[MAXFRAMECOUNT];
    IplImage *pTmpImage = NULL;
    char strImgPath[260] = {0};
    int nBlocks = (IMAGEWIDTH / nWinSize) * (IMAGEHEIGHT / nWinSize);
    BlockFlow *dbBlockFlow = new BlockFlow[nBlocks]; // xx, yy, alpha_angle
    memset(dbBlockFlow, 0, nBlocks * sizeof(BlockFlow));

    int last = 0;
    //读取图像序列
    for (int i = 0; (i < info.nLength) && (i < MAXFRAMECOUNT); i++) {
        if ((info.nStart + i) >= m_nVideoFrames || (info.nStart + i) < 0) {
            break;
        }

        sprintf(strImgPath, "%s/frame%d.jpg", m_imgPath, info.nStart + i);

        pTmpImage = cvLoadImage(strImgPath, 0);
        pSequenes[i] = cvCreateImage(cvSize(IMAGEWIDTH, IMAGEHEIGHT),
                pTmpImage->depth, pTmpImage->nChannels);
        cvResize(pTmpImage, pSequenes[i], 1);
        cvReleaseImage(&pTmpImage);

        last = i;
    }

    last++;

    PBlockFlow *pFrameFlow = new PBlockFlow[info.nLength - 1];
    //计算图像序列光流
    for (int i = 1; (i < last) && (i < info.nLength); i++) {
        ImgOpticalFlow(pSequenes[i - 1], pSequenes[i], nWinSize, dbBlockFlow);

        //求光流比较大的块
        pFrameFlow[i - 1] = new BlockFlow[MAXBLOCKCOUNT];
        memset(pFrameFlow[i - 1], 0, MAXBLOCKCOUNT * sizeof(BlockFlow));
        for (int m = 0; m < nBlocks; m++)
            for (int n = 0; n < MAXBLOCKCOUNT; n++) {
                if (abs(dbBlockFlow[m].xvalue) + abs(dbBlockFlow[m].yvalue) > abs(pFrameFlow[i - 1][n].xvalue) + abs(pFrameFlow[i - 1][n].yvalue)) {
                    if (n > 0) {
                        pFrameFlow[i - 1][n - 1] = pFrameFlow[i - 1][n];
                        pFrameFlow[i - 1][n] = dbBlockFlow[m];
                    }
                    else {
                        pFrameFlow[i - 1][0] = dbBlockFlow[m];
                    }
                }
            }
    } // for

    //释放图像序列
    for (int i = 0; (i < last) && (i < info.nLength); i++) {
        cvReleaseImage(&pSequenes[i]);
    }

    //序列比对
    double dbScore[MAXFRAMECOUNT][MAXFRAMECOUNT] = {0};
    double a = 0, b = 0, c = 0;
    double tmpSC = 0.0;
    int pairs = 0;
    double AlignRes = 0.0;

    for (int i = 1; i < m_vecGTInfo[m_nMaxSeqPos].nLength; i++)
        for (int j = 1; (j < last) && (j < info.nLength); j++) {
            tmpSC = SC_frames(m_SeqFlow[m_nMaxSeqPos][i - 1], pFrameFlow[j - 1]);
            a = dbScore[i][j - 1];
            b = dbScore[i - 1][j];
            c = dbScore[i - 1][j - 1] + tmpSC;
            dbScore[i][j] = max3(a, b, c);

            if (tmpSC > 0.3)
                pairs++;
        }

    //匹配对至少大于8
    if (pairs >= 2 || (pairs == 1 && dbScore[m_vecGTInfo[m_nMaxSeqPos].nLength - 1][info.nLength - 1] > m_dbAvgScore))
        AlignRes = dbScore[m_vecGTInfo[m_nMaxSeqPos].nLength - 1][info.nLength - 1] /*/pairs*/;

    delete[] dbBlockFlow;
    for (int i = 0; (i < last - 1) && (i < info.nLength - 1); i++) {
        delete[] pFrameFlow[i];
    }
    delete[] pFrameFlow;

    return AlignRes;
}

//根据回放镜头信息，修正镜头分割结果
int CReplayDetector::UpdateShotCutInfo(void) {
    int logoCount = (int) m_confirmLogo.size();
    int gradientCount = (int) m_vecAllGTInfo.size();
    int cutCount = (int) m_vecCutInfo.size();

    //渐变数小于3, 则直接输出所有的镜头数, 不再继续配对, 退出程序
    // if( gradientCount < 3 )
    if (gradientCount < 3 || m_nMaxSeqPos < 0) {
        printf("Gradual Shot Num:%d\n", gradientCount);
        printf("Shear Shot Num:%d\n", cutCount);
        m_vecUpdateGTInfo = m_vecAllGTInfo;
        m_vecUpdateCutInfo = m_vecCutInfo;

        for (int i = 0; i < (int) m_vecUpdateCutInfo.size(); i++)
            m_vecAllCut.push_back(m_vecUpdateCutInfo[i]);

        for (int i = 0; i < (int) m_vecUpdateGTInfo.size(); i++)
            m_vecAllCut.push_back(m_vecUpdateGTInfo[i].nStart);

        sort(m_vecAllCut.begin(), m_vecAllCut.end());
        printf("UpdateShotCutInfo--Shot Totalnum:%d\n", (int) m_vecAllCut.size());

        return 0;
    }

    ReplayInfo tmpReplay;
    GTInfo tmpInfo;
    tmpInfo.nLength = m_vecGTInfo[m_nMaxSeqPos].nLength;

    //配对logo成replay
    for (int i = 1; i < logoCount;) {
        if (m_confirmLogo[i] - m_confirmLogo[i - 1] > 40/*1600*/) { //1600帧对于斯诺克视频太大了, 调小一点
            //成功配对, 记录并跳过这两帧
            tmpInfo.nStart = m_confirmLogo[i - 1];
            tmpReplay.start = tmpInfo;
            tmpInfo.nStart = m_confirmLogo[i];
            tmpReplay.end = tmpInfo;
            m_vecReplay.push_back(tmpReplay); //tmpInfo的长度都是一样的, 都为m_vecGTInfo[m_nMaxSeqPos].nLength;
            i += 2;
        }
        else {
            //不成功配对，跳过一个
            i += 1;
        }
    }

    //没有检测到回放镜头，不进行修正
    if (m_vecReplay.size() == 0) {
        m_vecUpdateGTInfo = m_vecAllGTInfo;
        m_vecUpdateCutInfo = m_vecCutInfo;
    }
    else {
        //渐变处理
        for (int i = 0; i < (int) m_vecReplay.size(); i++) {
            m_vecUpdateGTInfo.push_back(m_vecReplay[i].start);
            m_vecUpdateGTInfo.push_back(m_vecReplay[i].end);
        }
        int j = 0;
        for (int i = 0; i < gradientCount; i++) {
            if (j < (int) m_vecReplay.size()) {
                if (m_vecAllGTInfo[i].nStart > m_vecReplay[j].start.nStart) {
                    if (m_vecAllGTInfo[i].nStart < m_vecReplay[j].end.nStart - 20 && m_vecAllGTInfo[i].nStart > m_vecReplay[j].start.nStart + 20) {
                        m_vecUpdateGTInfo.push_back(m_vecAllGTInfo[i]);
                    }
                }
                if (m_vecAllGTInfo[i].nStart > m_vecReplay[j].end.nStart) {
                    j++;
                    i--;
                    if (i < 0)
                        i = 0;
                }
            }
        }

        //切变处理
        int lastCut = -1;
        j = 0;
        for (int i = 0; i < cutCount; i++) {
            if (j < (int) m_vecReplay.size()) {
                //位于replay之间的切变是渐变
                if (m_vecCutInfo[i] < m_vecReplay[j].end.nStart - 20 && m_vecCutInfo[i] > m_vecReplay[j].start.nStart + 20) {
                    GTInfo tmpInfo;
                    tmpInfo.nStart = m_vecCutInfo[i] - 2;
                    tmpInfo.nLength = 8;
                    m_vecUpdateGTInfo.push_back(tmpInfo);
                }
                else {
                    //普通切变
                    if (lastCut != i) {
                        bool bNeed = true;
                        for (int k = 0; k < (int) m_vecReplay.size(); k++) {
                            if (abs(m_vecCutInfo[i] - m_vecReplay[k].start.nStart) < 20 || abs(m_vecCutInfo[i] - m_vecReplay[k].end.nStart) < 20)
                                bNeed = false;
                        }
                        if (bNeed) {
                            m_vecUpdateCutInfo.push_back(m_vecCutInfo[i]);
                        }
                    }
                    lastCut = i;
                }

                if (m_vecCutInfo[i] > m_vecReplay[j].end.nStart) {
                    j++;
                    i--;
                    if (i < 0)
                        i = 0;
                }
            }
            else {
                //普通切变
                if (lastCut != i && abs(m_vecCutInfo[i] - m_vecReplay[j - 1].end.nStart) > 20)
                    m_vecUpdateCutInfo.push_back(m_vecCutInfo[i]);

                lastCut = i;
            }
        }
        //渐变排序
        sort(m_vecUpdateGTInfo.begin(), m_vecUpdateGTInfo.end(), less_start);

    } // else

    for (int i = 0; i < (int) m_vecUpdateCutInfo.size(); i++)
        m_vecAllCut.push_back(m_vecUpdateCutInfo[i]);

    for (int i = 0; i < (int) m_vecUpdateGTInfo.size(); i++)
        m_vecAllCut.push_back(m_vecUpdateGTInfo[i].nStart);

    sort(m_vecAllCut.begin(), m_vecAllCut.end());

    return 1;
}

int CReplayDetector::SaveInfo(char *replaypath, char *replayFeatpath,
        char *allcutpath, char *gradientpath,
        char *cutpath) {
    FILE *freplay = NULL;
    FILE *freplayFeat = NULL;
    FILE *fallcut = NULL;
    FILE *fgradient = NULL;
    FILE *fcut = NULL;

    if (replaypath == NULL || replayFeatpath == NULL || allcutpath == NULL)
        return 0;

    freplay = fopen(replaypath, "wc");
    if (freplay == NULL)
        return 0;

    freplayFeat = fopen(replayFeatpath, "wc");
    if (freplayFeat == NULL)
        return 0;

    fallcut = fopen(allcutpath, "wc");
    if (fallcut == NULL)
        return 0;

    if (gradientpath == NULL)
        fgradient = fopen(m_gradientPath, "wc");
    else
        fgradient = fopen(gradientpath, "wc");

    if (fgradient == NULL)
        return 0;

    if (cutpath == NULL)
        fcut = fopen(m_cutPath, "wc");
    else
        fcut = fopen(cutpath, "wc");

    if (fcut == NULL)
        return 0;

    double alpha = 5.0;
    double Lconst = 2000.0;
    int replayLength = 0;
    double value = 0;
    int beginBound, endBound;
    double *replayFeat = (double *) malloc(m_nVideoFrames * sizeof(double));
    memset(replayFeat, 0, m_nVideoFrames * sizeof(double));
    int m_vecReplayLen = m_vecReplay.size();

    for (int i = 0; i < m_vecReplayLen; i++) {
        replayLength = m_vecReplay[i].end.nStart - m_vecReplay[i].start.nStart;
        beginBound = m_vecReplay[i].start.nStart - replayLength;
        endBound = m_vecReplay[i].end.nStart;
        if (beginBound < 0)
            beginBound = 0;
        for (int j = beginBound; j < endBound; j++) {
            value = replayLength / Lconst * exp(-(j - m_vecReplay[i].start.nStart) * (j - m_vecReplay[i].start.nStart) / (alpha * replayLength * replayLength));
            if (replayFeat[j] < value) {
                replayFeat[j] = value;
            }
        }
    }
    for (int i = 0; i < m_nVideoFrames; i++) {
        fprintf(freplayFeat, "%lf\n", replayFeat[i]);
    }
    for (int i = 0; i < m_vecReplayLen; i++) {
        fprintf(freplay, "%d	%d	%d	%d\n",
                m_vecReplay[i].start.nStart, m_vecReplay[i].start.nLength,
                m_vecReplay[i].end.nStart, m_vecReplay[i].end.nLength);
    }

    for (int i = 0; i < (int) m_vecAllCut.size(); i++) {
        fprintf(fallcut, "%d\n", m_vecAllCut[i]);
    }

    for (int i = 0; i < (int) m_vecUpdateGTInfo.size(); i++) {
        fprintf(fgradient, "%d	%d\n", m_vecUpdateGTInfo[i].nStart,
                m_vecUpdateGTInfo[i].nLength);
    }

    for (int i = 0; i < (int) m_vecUpdateCutInfo.size(); i++) {
        fprintf(fcut, "%d\n", m_vecUpdateCutInfo[i]);
    }

    free(replayFeat);
    fclose(freplay);
    fclose(freplayFeat);
    fclose(fallcut);
    fclose(fgradient);
    fclose(fcut);

    finish = clock();
    double totaltime = (double) (finish - start) / CLOCKS_PER_SEC;
    fprintf(stdout, "Replay Shot Detection Over, Use:%.02lf seconds!\n",
            totaltime);
    return 1;
}

/**
*  以渐变的数量和长度来判定是否包含回放片段
*
*  @return 是否包含回放片段
*/
bool CReplayDetector::isReplayExist() {
    //读取渐变
    FILE *fp = fopen(m_gradientPath, "r");
    GTInfo tmpInfo;
    if (fp == NULL) {
        printf("File is not exist");
        return 0;
    }
    while (fscanf(fp, "%d %d", &tmpInfo.nStart, &tmpInfo.nLength) != EOF) {
        m_vecAllGTInfo.push_back(tmpInfo);
        //渐变太短不要
        if (tmpInfo.nLength > 8 && m_vecGTInfo.size() < 60) {
            m_vecGTInfo.push_back(tmpInfo);
        }
    }
    fclose(fp);
    //如果渐变太少，则不适用此算法
    if (m_vecGTInfo.size() < 3) {
        printf("No Replay Shot -- Gradient shot must more than 2 .\n");
        return 0;
    }
    else {
        //分配空间，为回放序列比对做准备
        int nSCCount = static_cast<int>(m_vecGTInfo.size());
        m_scFrame = new SCSEQUENES *[nSCCount];
        for (int i = 0; i < nSCCount; i++) {
            m_scFrame[i] = new SCSEQUENES[nSCCount];
            for (int j = 0; j < nSCCount; j++)
                m_scFrame[i][j] = 0;
        }
        return 1;
    }
}
