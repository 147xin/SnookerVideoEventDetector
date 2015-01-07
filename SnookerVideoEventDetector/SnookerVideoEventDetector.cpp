//
//  SnookerVideoEventDetector.cpp
//  SnookerVideoEventDetector
//
//  Created by Yixin Huang on 14/12/22.
//  Copyright (c) 2014年 Yixin Huang. All rights reserved.
//

#include "SnookerVideoEventDetector.h"

using namespace std;
using namespace cv;

//#define DEBUG_FULL_TABLE_VIEW_DETECTION
//#define DEBUG_SCORE_BAR_DETECTION
#define DEBUG_TURN_INDICATOR

#define GREEN_THRESHOLD 20


void SnookerVideoEventDetector::SetVideoFilePath(const std::string &videoPath) {
    videoInfo.videoFilePath = videoPath;
}

/**
*  设置帧图像存放文件夹, 并获取视频帧率和总帧数
*
*  @param framesPath 用于存放帧图像的文件夹(包含最后的左斜线)
*/
void SnookerVideoEventDetector::GetVideoFrames(const std::string &framesPath) {
    videoInfo.framesFolder = framesPath;
    unsigned int frameNum = 0;
    if (videoInfo.videoFilePath[videoInfo.videoFilePath.length() - 1] != '/') {
        VideoCapture capture(videoInfo.videoFilePath);
        if (!capture.isOpened()) {
            cout << "Cannot open video file!" << endl;
            exit(1);
        }
        videoInfo.fps = capture.get(CV_CAP_PROP_FPS);
        Mat frame;
        while (capture.read(frame)) {
            if (videoInfo.width == 0 || videoInfo.height == 0) {
                videoInfo.width = frame.cols;
                videoInfo.height = frame.rows;
            }
            string picPath(videoInfo.framesFolder);
            picPath += "frame" + to_string(frameNum) + ".jpg";
            imwrite(picPath, frame);
            cout << "Write frame " << frameNum << "\n";
            ++frameNum;
        }
        capture.release();
    }
    else {
        int segNum = 1;
        string segPath = videoInfo.videoFilePath + to_string(segNum) + ".mp4";
        VideoCapture capture(segPath);
        if (!capture.isOpened()) {
            cout << "Cannot open video file!" << endl;
            exit(1);
        }

        while (capture.isOpened()) {
            videoInfo.fps = capture.get(CV_CAP_PROP_FPS);
            Mat frame;
            while (capture.read(frame)) {
                string picPath(videoInfo.framesFolder);
                picPath += "frame" + to_string(frameNum) + ".jpg";
                imwrite(picPath, frame);
                cout << "Write frame " << frameNum << "\n";
                ++frameNum;
            }

            ++segNum;
            string segPath = videoInfo.videoFilePath + to_string(segNum) + ".mp4";
            capture.open(segPath);//!!!这里会输出警告信息, 待今后完善
        }
        capture.release();
    }
    videoInfo.framesNum = frameNum;
}

void SnookerVideoEventDetector::SetReplayDetectorOutputPath(
        const std::string &outputPath) {
    videoInfo.cutPath = outputPath + "cutResult.txt";
    videoInfo.gtPath = outputPath + "gradientResult.txt";
    videoInfo.allCutPath = outputPath + "allCut.txt";
    videoInfo.cutUpdatePath = outputPath + "cutUpdate.txt";
    videoInfo.gtUpdatePath = outputPath + "gtUpdate.txt";
    videoInfo.replayPath = outputPath + "replay.txt";
    videoInfo.replayFeatPath = outputPath + "replayFeat.txt";
}

void SnookerVideoEventDetector::GetReplayInfo() {
    //切变与渐变的检测
    CShotCut shotcut;
    shotcut.SetPath((char *) videoInfo.framesFolder.c_str(),
            videoInfo.framesNum);
    shotcut.ShotDetection();
    shotcut.SaveInfo((char *) videoInfo.cutPath.c_str(),
            (char *) videoInfo.gtPath.c_str());

    //回放检测
    CReplayDetector replayDetector;

    //设置渐变/切变/帧图像文件夹路径
    replayDetector.SetPath((char *) videoInfo.gtPath.c_str(),
            (char *) videoInfo.cutPath.c_str(),
            (char *) videoInfo.framesFolder.c_str());
    //设置视频总帧数
    replayDetector.SetVideoFrames(videoInfo.framesNum);
    //读取切变信息, 渐变信息在isReplayExist函数中读取
    replayDetector.ReadFile();

    bool isReplayExist = false;
    if ((isReplayExist = replayDetector.isReplayExist())) {
        cout << "Replay exist!\n";
        //计算光流
        replayDetector.CalcOpticalFlow();
        //获取 Logo Template
        replayDetector.GetLogoTemplate();
        //获取 L-pixels(Logo-pixels)
        if (replayDetector.GetLpixels() != 0) {
            replayDetector.LogoDetection();
        }
    }
    replayDetector.UpdateShotCutInfo();
    replayDetector.SaveInfo(
            (char *) videoInfo.replayPath.c_str(), (char *) videoInfo.replayFeatPath.c_str(),
            (char *) videoInfo.allCutPath.c_str(), (char *) videoInfo.gtUpdatePath.c_str(),
            (char *) videoInfo.cutUpdatePath.c_str());


    cout << "Replay detection complete!" << endl;
}

void SnookerVideoEventDetector::ReadReplayInfo() {
    //将回放信息记录到类的视频信息中
    ifstream ifs(videoInfo.replayPath);
    string line;
    while (getline(ifs, line) && !boost::trim_copy(line).empty()) {
        stringstream lineStream(line);
        int start, length;
        ReplayInfo replayInfo;
        lineStream >> start >> length;
        replayInfo.start.nStart = start;
        replayInfo.start.nLength = length;
        lineStream >> start >> length;
        replayInfo.end.nStart = start;
        replayInfo.end.nLength = length;
        videoInfo.replays.push_back(replayInfo);
    }
    //构造判断某一帧是否是回放镜头帧的哈希表
    for (vector<ReplayInfo>::const_iterator it = videoInfo.replays.cbegin(); it != videoInfo.replays.cend(); ++it) {
        for (int frameNum = (*it).start.nStart; frameNum != (*it).start.nStart + (*it).start.nLength; ++frameNum) {
            videoInfo.replayCheckHashTab.insert((unsigned int) frameNum);
        }
        for (int frameNum = (*it).end.nStart; frameNum != (*it).end.nStart + (*it).start.nLength; ++frameNum) {
            videoInfo.replayCheckHashTab.insert((unsigned int) frameNum);
        }
    }
}

void SnookerVideoEventDetector::GetFrameFeature(int const currentFrameNum, FrameFeature &frameFeature) {
    frameFeature.frameId = currentFrameNum;
    //读取该帧图像
    string imgPath(videoInfo.framesFolder);
    imgPath += "frame" + to_string(currentFrameNum) + ".jpg";
    Mat srcImg = imread(imgPath);
//#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    //imshow("Image", srcImg);
    //waitKey(0);
//#endif

    //创建用于存储台面区域的图像
    Mat maskImg(srcImg.rows, srcImg.cols, CV_8U);

    //记录开始时间
    // double duration = static_cast<double>(getTickCount());

    //检测台面区域
    Mat_<Vec3b>::iterator itSrc = srcImg.begin<Vec3b>();
    Mat_<uchar>::iterator itMask = maskImg.begin<uchar>();
    for (; itSrc != srcImg.end<Vec3b>(); ++itSrc, ++itMask) {
        if ((*itSrc)[1] - (*itSrc)[0] > GREEN_THRESHOLD &&
                (*itSrc)[1] - (*itSrc)[2] > GREEN_THRESHOLD) {
            *itMask = 255;
        } else {
            *itMask = 0;
        }
    }
#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    imshow("Image", maskImg);
    waitKey(0);
#endif
    //膨胀，去除小物体，比如球杆
    RemoveSmallObjects(maskImg, Size(srcImg.cols, srcImg.rows));
#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    imshow("Image", maskImg);
    waitKey(0);
#endif
    //记录耗费时间
    // duration = static_cast<double>(getTickCount() - duration) /
    // getTickFrequency();
    // cout << duration << "s" << endl;

    //对桌面区域图像进行Canny边缘检测
    Mat contours(srcImg.rows, srcImg.cols, CV_8U);
    Canny(maskImg, contours, 125, 350);
#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    imshow("Image", contours);
    waitKey(0);
#endif

    //用霍夫变换检测直线
    vector<Vec2f> lines;
    HoughLines(contours, lines, 1, CV_PI / 180, 70);

#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    //复制原图像一份，用于显示直线检测结果
    Mat srcImgCpy = srcImg.clone();
    DrawDetectedLines(srcImgCpy, lines, Scalar(255, 255, 255));
    imshow("Image", srcImgCpy);
    waitKey(0);
#endif
    //移除相近的直线
    RemoveNearbyLines(lines, Size(srcImg.cols, srcImg.rows));

#ifdef DEBUG_FULL_TABLE_VIEW_DETECTION
    //画出相近直线移除之后的直线
    DrawDetectedLines(srcImgCpy, lines, Scalar(0, 0, 255), 2);
    imshow("Image", srcImgCpy);
    waitKey(0);
#endif
    //根据直线的位置和角度信息判定是否是全台面视角
    bool isFullTableView =
            IsFullTableView(lines, Size(srcImg.cols, srcImg.rows));
    frameFeature.isFullTable = isFullTableView;
    cout << "Frame " << currentFrameNum << " is" << (isFullTableView ? "" : " NOT")
            << " full table view." << endl;

    //如果是全台面视角，则进行比分条提取
    if (isFullTableView) {
        //如果是全台面视角, 并且之前没有确定过比分条区域, 则开始寻找比分条区域
        if (videoInfo.scorebarRegion.x == 0) { //还没有确定比分条区域
            //求直线的交点
            int lineNum = static_cast<int>(lines.size());
            vector<Point> points; //存放直线的交点集合
            for (int i = 0; i < lineNum; ++i) {
                for (int j = i + 1; j < lineNum; ++j) {
                    IntersectionPoint(lines[i], lines[j], points, srcImg.rows,
                            srcImg.cols);
                }
            }

#ifdef DEBUG_SCORE_BAR_DETECTION
            //将交点显示在原图上
            DrawCircles(srcImgCpy, points, Scalar(255, 255, 255));
            imshow("Image", srcImgCpy);
            waitKey(0);
#endif

            //在交点集合中选取离左右两边缘最近的一个点
            Point myPoint;
            bool leftSide;
            SelectPoint(points, Size(srcImg.cols, srcImg.rows), myPoint, leftSide);
#ifdef DEBUG_SCORE_BAR_DETECTION
            //显示所选取的点
            circle(srcImgCpy, myPoint, 5, Scalar(173, 255, 47), 3);
            imshow("Image", srcImgCpy);
            waitKey(0);
#endif
            //估计比分条区域的高度
            // int barHeight = TableRegionHeight(points, srcImg.rows) * 0.15;
            int barHeight = static_cast<int>(srcImg.rows * 0.08);
            //截取比分条区域图像
            int barWidth;
            if (!leftSide) {
                myPoint.x = srcImg.cols - myPoint.x;
            }
            barWidth = srcImg.cols - 2 * myPoint.x;
            Size barSize(barWidth, barHeight);
            Mat scorebar = srcImg(Rect(
                    myPoint.x, myPoint.y + static_cast<int>(TableRegionHeight(points, srcImg.rows) * 0.03),
                    barWidth, barHeight));
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Original Score Bar Region", scorebar);
            waitKey(0);
#endif
            //将比分条区域灰度化
            Mat scorebarCpy(barSize.height, barSize.width, CV_8U);
            cvtColor(scorebar, scorebarCpy, CV_BGR2GRAY, 1);
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar Region", scorebarCpy);
            waitKey(0);
#endif
            //进行阈值处理
            //            threshold(scorebarCpy, scorebarCpy, 100, 255,
            //            THRESH_BINARY);
            //            imshow("Score Bar Region", scorebarCpy);
            //            waitKey(0);
            //这里要不要加个对比度拉伸???有强烈的诉求！为避免后续比分条垂直分割的困难!!!???
            //int i = 0;

            //对比分条区域进行边缘检测
            Mat barContours(barSize.height, barSize.width, CV_8U);
            Canny(
                    scorebarCpy, barContours, 200 /*125*/,
                    350); //这里边缘检测有点不准确！比如三角形的下边缘丢失！关于边缘检测两个阈值的含义，好好看看图像处理！
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar Region", barContours);
            waitKey(0);
#endif
            //除去边缘图像中的长水平与垂直线
            //对水平与垂直方向的梯度图像进行以直线型的开操作以获得长直线区域
            //再将原图像与该结果相减以去除长直线
            Mat lineKernelX(1, (int) round(barHeight * 0.3), CV_8U, Scalar(255));
            Mat lineKernelY((int) round(barHeight * 0.3), 1, CV_8U, Scalar(255));
            Mat lineX(barSize.height, barSize.width, CV_8U);
            Mat lineY(barSize.height, barSize.width, CV_8U);

            erode(barContours, lineX, lineKernelX);
            dilate(lineX, lineX, lineKernelX);
            //                        imshow("LineX", lineX);
            //                        waitKey(0);

            //暂时未使用垂直方向的长直线
            //                erode(barContours, lineY, lineKernelY);
            //                dilate(lineY, lineY, lineKernelY);
            //                        imshow("LineY", lineY);
            //                        waitKey(0);

            Mat barContoursCpy = barContours - lineX; //暂时不减去垂直长直线
            //      imshow("Score Bar Region", barContoursCpy);
            //      waitKey(0);

            //用Sobel算子进行梯度强度检测
            Mat sobelX(barSize.height, barSize.width, CV_16S);
            Mat sobelY(barSize.height, barSize.width, CV_16S);
            Sobel(barContoursCpy, sobelX, CV_16S, 1, 0, 3);
            Sobel(barContoursCpy, sobelY, CV_16S, 0, 1, 3);
            Mat sobel = abs(sobelX) + abs(sobelY);
            double sobmin, sobmax;
            minMaxLoc(sobel, &sobmin, &sobmax);
            sobel.convertTo(sobel, CV_8U, 255.0 / sobmax, 0);

            //      imshow("Score Bar Region", sobel);
            //      waitKey(0);

            //使用对角方向的Sobel算子进行梯度计算
            //            Mat sobKernel1(3,3,CV_8S,Scalar(0));
            //            sobKernel1.at<char>(0,0)=0;
            //            sobKernel1.at<char>(0,1)=1;
            //            sobKernel1.at<char>(0,2)=2;
            //            sobKernel1.at<char>(1,0)=-1;
            //            sobKernel1.at<char>(1,1)=0;
            //            sobKernel1.at<char>(1,2)=1;
            //            sobKernel1.at<char>(2,0)=-2;
            //            sobKernel1.at<char>(2,1)=-1;
            //            sobKernel1.at<char>(2,2)=0;
            ////            imshow("Kernel", sobKernel1);
            ////            waitKey(0);
            //            Mat sobKernel2(3,3,CV_8S,Scalar(0));
            //            sobKernel2.at<char>(0,0)=-2;
            //            sobKernel2.at<char>(0,1)=-1;
            //            sobKernel2.at<char>(0,2)=0;
            //            sobKernel2.at<char>(1,0)=-1;
            //            sobKernel2.at<char>(1,1)=0;
            //            sobKernel2.at<char>(1,2)=1;
            //            sobKernel2.at<char>(2,0)=0;
            //            sobKernel2.at<char>(2,1)=1;
            //            sobKernel2.at<char>(2,2)=2;
            //            imshow("Kernel", sobKernel2);
            //            waitKey(0);
            //            filter2D(barContours, sobelX, scorebarCpy.depth(),
            //            sobKernel1);
            //            filter2D(barContours, sobelY, scorebarCpy.depth(),
            //            sobKernel2);

            //            Mat sobeld(barSize.height, barSize.width, CV_16U);
            //            sobeld = abs(sobelXd) + abs(sobelYd);
            //#ifdef DEBUG_SCORE_BAR_DETECTION
            //            imshow("Score Bar Region", sobeld);
            //            waitKey(0);
            //#endif
            //搜寻sobel中的极大值
            //            double sobmin, sobmax;
            //            minMaxLoc(sobel, &sobmin, &sobmax);
            //            Mat sobelImage;
            //            sobel.convertTo(sobelImage, CV_8U, -255.0 / sobmax, 255);
            //            imshow("Score Bar Region", sobelImage);
            //            waitKey(0);

            //确定文字所在的水平区域
            //统计边缘梯度图像在水平方向的投影值数组
            vector<int> x(barSize.height, 0);
            int sum = 0;
            for (int i = 0; i < barSize.height; ++i) {
                uchar *row = sobel.ptr<uchar>(i);
                for (int j = 0; j < barSize.width; ++j) {
                    //                    if (row[j]) {
                    //                        ++x[i];
                    //                        ++sum;
                    //                    }
                    x[i] += row[j];
                    sum += row[j];
                }
            }
            //测试，查看投影数组的值

            //求均值
            int avg = sum / barSize.height;
            //求最大值
            int maxValue = 0;
            for (int i = 0; i < x.size(); ++i) {
                if (x[i] > maxValue) {
                    maxValue = x[i];
                }
            }
            //如果最大值小于一个阈值，说明这个区域并没有比分条
            if (avg * 1.0 < barSize.width * 10) {
                cout << "Frame " << currentFrameNum << " has no score bar!" << endl;
                frameFeature.hasScoreBar = false;
                videoInfo.frameFeatures.push_back(frameFeature);
                return;
            }
            //frameFeature.hasScoreBar=true; //这个时候还不能判定是否含有比分条, 因为是否含有合法的字段还无法获知
            //因为有可能是比分条的动画还没结束

            //确定更精确的比分条水平范围
            Mat scorebar1;
            int thresh = maxValue * 0.7;
            //从上到下找到第一个大于均值的位置，从下往上找到第一个大于均值的位置
            int upbound = 0, downbound = 0;
            for (int idx = 0; idx < barSize.height; ++idx) {
                if (x[idx] > thresh) {
                    upbound = idx;
                    break;
                }
            }
            for (int idx = barSize.height - 1; idx >= 0; --idx) {
                if (x[idx] > thresh) {
                    downbound = idx;
                    break;
                }
            }
            upbound -= round(barHeight * 0.1);
            // upbound-=2;
            if (upbound < 0)
                upbound = 0;
            downbound += round(barHeight * 0.1);
            // downbound+=2;
            if (downbound > barHeight - 1) {
                downbound = barHeight - 1;
            }
            //缩小比分条的高度
            barHeight = downbound - upbound + 1;
            scorebar1 = scorebar(Rect(0, upbound, barWidth, barHeight));

            //                    //如果之前未记录比分条区域的位置, 则进行记录
            //                    if (videoInfo.scorebarRegion.x == 0) {
            //                        videoInfo.scorebarRegion.x = myPoint.x;
            //                        videoInfo.scorebarRegion.y = myPoint.y + TableRegionHeight(points, srcImg.rows) * 0.03 + upbound;
            //                        videoInfo.scorebarRegion.width = barWidth;
            //                        videoInfo.scorebarRegion.height = barHeight;
            //                    }
            //记录可能的比分条区域
            Rect tmpScorebarRegion;//可能的比分条区域, 但不一定是的, 因为可能是比分条渐变过程中的一个动画状态
            tmpScorebarRegion.x = myPoint.x;
            tmpScorebarRegion.y = myPoint.y + TableRegionHeight(points, srcImg.rows) * 0.03 + upbound;
            tmpScorebarRegion.width = barWidth;
            tmpScorebarRegion.height = barHeight;
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar", scorebar1);
            waitKey(0);
#endif


            //已知比分条区域, 开始进行当前击球球员指示符位置识别
            //Mat scorebar=srcImg(tmpScorebarRegion);

            //缩小以后的边缘图像
            Mat barContoursNew =
                    barContoursCpy(Rect(0, upbound, barWidth, barHeight));
            //      imshow("Contours", barContoursNew);
            //      waitKey(0);
            //            Mat barContoursDilated;
            //            int seSize = barHeight * 0.3;
            //            Mat se(seSize, seSize, CV_8U, Scalar(1));
            //            Mat se1(seSize+1, seSize+1, CV_8U, Scalar(1));
            //            dilate(barContoursNew, barContoursDilated, se);
            //            erode(barContoursDilated, barContoursDilated, se1);
            //            imshow("Score Bar Region", barContoursDilated);
            //            waitKey(0);

            //用洪泛法填充与边缘连接的区域，比如长竖线
            uchar *row = barContoursNew.ptr<uchar>(0);
            for (int i = 0; i < barSize.width; ++i) {
                if (row[i] == 255) {
                    floodFill(barContoursNew, Point(i, 0), 0);
                }
            }
            row = barContoursNew.ptr<uchar>(barHeight - 1);
            for (int i = 0; i < barSize.width; ++i) {
                if (row[i] == 255) {
                    floodFill(barContoursNew, Point(i, barHeight - 1), 0);
                }
            }
            //            imshow("Score Bar Region", barContoursNew);
            //            waitKey(0);
            int lineSELen = round(barHeight * 0.5);
            //保证长度是奇数
            if (lineSELen % 2 == 0) {
                ++lineSELen;
            }
            Mat lineSE(1, lineSELen, CV_8U, Scalar(0));
            lineSE.ptr<uchar>(0)[lineSELen / 2 + 1] = 255;
            //      imshow("lineSE", lineSE);
            //      waitKey(0);
            //对边界图像进行模板匹配，找出孤立的竖直线
            Mat matchResult;
            matchTemplate(barContoursNew, lineSE, matchResult, CV_TM_SQDIFF);
            //      imshow("matchResult", matchResult);
            //      waitKey(0);
            matchResult.convertTo(matchResult, CV_8U);
            matchResult.convertTo(matchResult, CV_8U, -1, 255);
            //      imshow("matchResult", matchResult);
            //      waitKey(0);
            //上面的直线匹配图像比原边缘图像略小，相减之前需要取得原边缘图像的子区域
            Mat barContoursNewRoi = barContoursNew(
                    Rect(lineSELen / 2 + 1, 0, barWidth - lineSELen + 1, barHeight));
            //      imshow("barContoursNewRoi", barContoursNewRoi);
            //      waitKey(0);

            // Mat test(barHeight, barWidth-lineSELen+1, CV_8U,Scalar(255));

            // barContoursNewRoi.convertTo(barContoursNewRoi,-1,-1,255);
            barContoursNewRoi = barContoursNewRoi - matchResult;
            //      imshow("Contours", barContoursNew);
            //      waitKey(0);
            //            Mat barContoursNewEroded(barSize.height, barSize.width,
            //            CV_8U);
            //            erode(barContoursNew, barContoursNewEroded, lineSE);
            //            imshow("barContoursNewEroded", barContoursNewEroded);
            //            waitKey(0);
            //            barContoursNew = barContoursNew-barContoursNewEroded;
            //            imshow("barContoursNew", barContoursNew);
            //            waitKey(0);

            //用Sobel算子进行梯度强度检测
            Sobel(barContoursNew, sobelX, CV_16S, 1, 0, 3);
            Sobel(barContoursNew, sobelY, CV_16S, 0, 1, 3);
            Mat sobel1 = abs(sobelX) + abs(sobelY);
            minMaxLoc(sobel1, 0, &sobmax);
            sobel1.convertTo(sobel1, CV_8U, 255.0 / sobmax, 0);

            //      imshow("Contours", sobel1);
            //      waitKey(0);

            //统计边缘梯度图像在垂直方向上的投影值
            vector<int> verticalSums(barWidth, 0);
            sum = 0;
            for (int i = 0; i < barHeight; ++i) {
                uchar *row = sobel1.ptr<uchar>(i);
                for (int j = 0; j < barWidth; ++j) {
                    //                    if (row[j]) {
                    //                        ++x[i];
                    //                        ++sum;
                    //                    }
                    verticalSums[j] += row[j];
                    sum += row[j];
                }
            }
            cout << "Frame " << currentFrameNum << " vertical sums:" << endl;
            for (int value : verticalSums) {
                cout << value << endl;
            }
            //均值
            int verticalSumsAvg = sum / barWidth;
            cout << "average: " << verticalSumsAvg << endl;
            //将小于阈值的值置为0
            int cutThreshold = 2 * 255;
            for (int &value : verticalSums) {
                if (value < cutThreshold) {
                    value = 0;
                }
            }
            cout << "Frame " << currentFrameNum << " modified vertical sums:" << endl;
            for (int value : verticalSums) {
                cout << value << endl;
            }

            //判定当前击球球员指示符在哪一侧
            int leftSum = 0, rightSum = 0;
            for (int i = videoInfo.currentPlayerFlagPos[0][0]; i <= videoInfo.currentPlayerFlagPos[0][1]; ++i) {
                leftSum += verticalSums[i];
            }
            for (int i = videoInfo.currentPlayerFlagPos[1][0]; i <= videoInfo.currentPlayerFlagPos[1][1]; ++i) {
                rightSum += verticalSums[i];
            }
            if (leftSum > rightSum) {
                frameFeature.turn = 0;
            }
            else if (leftSum < rightSum) {
                frameFeature.turn = 1;
            }
            else {
                frameFeature.turn = -1;
            }
            //将指示符处的值置为0
            for (int i = videoInfo.currentPlayerFlagPos[0][0]; i <= videoInfo.currentPlayerFlagPos[0][1]; ++i) {
                verticalSums[i] = 0;
            }
            for (int i = videoInfo.currentPlayerFlagPos[1][0]; i <= videoInfo.currentPlayerFlagPos[1][1]; ++i) {
                verticalSums[i] = 0;
            }
            //开始垂直分割图像
            int gapThreshold = (int) (barHeight * 0.1);       //文字间隔阈值
            int textWidthThreshold = (int) (barHeight * 0.3); //文字宽度阈值
            vector<bool> textIndicator((unsigned long) barWidth, false);
            for (int i = 0; i < barWidth - textWidthThreshold + 1; ++i) {
                int zeroCount = 0;
                for (int j = i; j < i + textWidthThreshold; ++j) {
                    if (verticalSums[j] == 0) {
                        ++zeroCount;
                    }
                }
                if (zeroCount <= gapThreshold) {
                    for (int k = i + 1; k < i + textWidthThreshold - 1; ++k) {
                        textIndicator[k] = true;
                    }
                }
            }

            //填补中间的总局数左右括号与局数之间可能产生的大间距
            //将填补区域限定在 中点-barHeight/2~中点+barHeight/2
            //      int frameGapFillThreshold=gapThreshold*2;
            //      int frameWidth=round(barWidth/barHeight*barHeight*1);
            //      int frameGapFillEnd=barWidth/2+frameWidth/2-frameGapFillThreshold;
            //      for (int i=barWidth/2-frameWidth/2; i<frameGapFillEnd; ++i) {
            //        if (textIndicator[i]&&textIndicator[i+frameGapFillThreshold-1])
            //        {
            //          for (int j=i+1; j<i+frameGapFillThreshold-1; ++j) {
            //            textIndicator[j]=true;
            //          }
            //        }
            //      }

            //将中间1个barHeight宽度的地方填充，确保总局分不间断
            //      int fillMaxFramesGapBegin=barWidth/2-round(barHeight*0.4);
            //      int fillMaxFramesGapEnd=barWidth/2+round(barHeight*0.4);
            //      for (int i=fillMaxFramesGapBegin; i<=fillMaxFramesGapEnd; ++i) {
            //        textIndicator[i]=true;
            //      }

            //为防止球员姓名之间的间距过大导致姓名断开，将比分条左右半边大于一定阈值的空隙进行填补
            //将填补区域限定在左边1/4和右边1/4
            int nameGapFillThreshold = gapThreshold * 3;
            int leftEnd = barWidth / 4 - nameGapFillThreshold;
            int rightEnd = barWidth - nameGapFillThreshold;
            for (int i = 0; i < leftEnd; ++i) {
                if (textIndicator[i] && textIndicator[i + nameGapFillThreshold - 1]) {
                    for (int j = i + 1; j < i + nameGapFillThreshold - 1; ++j) {
                        textIndicator[j] = true;
                    }
                }
            }
            for (int i = barWidth * 3 / 4; i < rightEnd; ++i) {
                if (textIndicator[i] && textIndicator[i + nameGapFillThreshold - 1]) {
                    for (int j = i + 1; j < i + nameGapFillThreshold - 1; ++j) {
                        textIndicator[j] = true;
                    }
                }
            }

            //测试结果
            cout << "Frame " << currentFrameNum << " text areas:" << endl;
            for (bool value : textIndicator) {
                cout << (value ? "#####" : "0") << endl;
            }

            //确定各个字段的位置
            int maxFrameNumPos[2] = {0, 0};
            int frameNum1Pos[2] = {0, 0};
            int frameNum2Pos[2] = {0, 0};
            int score1Pos[2] = {0, 0};
            int score2Pos[2] = {0, 0};
            int name1Pos[2] = {0, 0};
            int name2Pos[2] = {0, 0};
            int name1PosPlanB[2] = {0, 0};
            int name2PosPlanB[2] = {0, 0};

            //向左检测
            int cnt = 0;
            for (int i = barWidth / 2; i >= 0; --i) {
                if (textIndicator[i] == false && textIndicator[i + 1] == true) {
                    if (cnt == 0) {
                        maxFrameNumPos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 2) {
                        frameNum1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 4) {
                        score1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 6) {
                        name1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 8) {
                        name1PosPlanB[0] = i + 1;
                        break;
                    }
                }
                if (textIndicator[i] == true && textIndicator[i + 1] == false) {
                    if (cnt == 1) {
                        frameNum1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 3) {
                        score1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 5) {
                        name1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 7) {
                        name1PosPlanB[1] = i;
                        ++cnt;
                    }
                }
            }
            //向右检测
            cnt = 0;
            for (int i = barWidth / 2; i < barWidth; ++i) {
                if (textIndicator[i] == false && textIndicator[i - 1] == true) {
                    if (cnt == 0) {
                        maxFrameNumPos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 2) {
                        frameNum2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 4) {
                        score2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 6) {
                        name2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 8) {
                        name2PosPlanB[1] = i - 1;
                        break;
                    }
                }
                if (textIndicator[i] == true && textIndicator[i - 1] == false) {
                    if (cnt == 1) {
                        frameNum2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 3) {
                        score2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 5) {
                        name2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 7) {
                        name2PosPlanB[0] = i;
                        ++cnt;
                    }
                }
            }

            //如果两个名字区域的宽度都小于一个阈值，则启用namePosPlanB
            int nameWidthThreshold = (int) round(barHeight * 1.2);
            if (name1Pos[1] - name1Pos[0] < nameWidthThreshold && name2Pos[1] - name2Pos[0] < nameWidthThreshold) {
                name1Pos[0] = name1PosPlanB[0];
                name1Pos[1] = name1PosPlanB[1];
                name2Pos[0] = name2PosPlanB[0];
                name2Pos[1] = name2PosPlanB[1];
            }

            //在原图上画线标示识别出的文字区域
            rectangle(scorebar1, Point(maxFrameNumPos[0], 0),
                    Point(maxFrameNumPos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(frameNum1Pos[0], 0),
                    Point(frameNum1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(frameNum2Pos[0], 0),
                    Point(frameNum2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(score1Pos[0], 0),
                    Point(score1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(score2Pos[0], 0),
                    Point(score2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(name1Pos[0], 0),
                    Point(name1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
            rectangle(scorebar1, Point(name2Pos[0], 0),
                    Point(name2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar", scorebar1);
            waitKey(0);
#endif
            //如果这些区域都存在的话, 则保存此时的比分条区域
            if (maxFrameNumPos[0] && maxFrameNumPos[1]
                    && frameNum1Pos[0] && frameNum1Pos[1]
                    && frameNum2Pos[0] && frameNum2Pos[1]
                    && score1Pos[0] && score1Pos[1]
                    && score2Pos[0] && score2Pos[1]
                    && name1Pos[0] && name1Pos[1]
                    && name2Pos[0] && name2Pos[1]) {
                frameFeature.maxFrameNumPos[0] = maxFrameNumPos[0];
                frameFeature.maxFrameNumPos[1] = maxFrameNumPos[1];
                frameFeature.frameNum1Pos[0] = frameNum1Pos[0];
                frameFeature.frameNum1Pos[1] = frameNum1Pos[1];
                frameFeature.frameNum2Pos[0] = frameNum2Pos[0];
                frameFeature.frameNum2Pos[1] = frameNum2Pos[1];
                frameFeature.score1Pos[0] = score1Pos[0];
                frameFeature.score1Pos[1] = score1Pos[1];
                frameFeature.score2Pos[0] = score2Pos[0];
                frameFeature.score2Pos[1] = score2Pos[1];
                frameFeature.name1Pos[0] = name1Pos[0];
                frameFeature.name1Pos[1] = name1Pos[1];
                frameFeature.name2Pos[0] = name2Pos[0];
                frameFeature.name2Pos[1] = name2Pos[1];
                frameFeature.scorebarRegion = tmpScorebarRegion;
                frameFeature.hasScoreBar = true;
            }
        }
        else {     //已经确定了比分条区域
            Mat scorebar = srcImg(videoInfo.scorebarRegion);
            //将比分条区域灰度化
            Mat scorebarCpy(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_8U);
            cvtColor(scorebar, scorebarCpy, CV_BGR2GRAY, 1);
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar Region", scorebarCpy);
            waitKey(0);
#endif
            //对比分条区域进行边缘检测
            Mat barContours(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_8U);
            Canny(
                    scorebarCpy, barContours, 200 /*125*/,
                    350); //这里边缘检测有点不准确！比如三角形的下边缘丢失！关于边缘检测两个阈值的含义，好好看看图像处理！
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar Region", barContours);
            waitKey(0);
#endif
            //除去边缘图像中的长水平与垂直线
            //对水平与垂直方向的梯度图像进行以直线型的开操作以获得长直线区域
            //再将原图像与该结果相减以去除长直线
            Mat lineKernelX(1, (int) round(videoInfo.scorebarRegion.height * 0.3), CV_8U, Scalar(255));
            Mat lineKernelY((int) round(videoInfo.scorebarRegion.height * 0.3), 1, CV_8U, Scalar(255));
            Mat lineX(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_8U);
            Mat lineY(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_8U);

            erode(barContours, lineX, lineKernelX);
            dilate(lineX, lineX, lineKernelX);


            Mat barContoursNew = barContours - lineX; //暂时不减去垂直长直线
            //      imshow("Score Bar Region", barContoursCpy);
            //      waitKey(0);

            //用洪泛法填充与边缘连接的区域，比如长竖线
            uchar *row = barContoursNew.ptr<uchar>(0);
            for (int i = 0; i < videoInfo.scorebarRegion.width; ++i) {
                if (row[i] == 255) {
                    floodFill(barContoursNew, Point(i, 0), 0);
                }
            }
            row = barContoursNew.ptr<uchar>(videoInfo.scorebarRegion.height - 1);
            for (int i = 0; i < videoInfo.scorebarRegion.width; ++i) {
                if (row[i] == 255) {
                    floodFill(barContoursNew, Point(i, videoInfo.scorebarRegion.height - 1), 0);
                }
            }
            //            imshow("Score Bar Region", barContoursNew);
            //            waitKey(0);
            int lineSELen = (int) round(videoInfo.scorebarRegion.height * 0.5);
            //保证长度是奇数
            if (lineSELen % 2 == 0) {
                ++lineSELen;
            }
            Mat lineSE(1, lineSELen, CV_8U, Scalar(0));
            lineSE.ptr<uchar>(0)[lineSELen / 2 + 1] = 255;
            //      imshow("lineSE", lineSE);
            //      waitKey(0);
            //对边界图像进行模板匹配，找出孤立的竖直线
            Mat matchResult;
            matchTemplate(barContoursNew, lineSE, matchResult, CV_TM_SQDIFF);
            //      imshow("matchResult", matchResult);
            //      waitKey(0);
            matchResult.convertTo(matchResult, CV_8U);
            matchResult.convertTo(matchResult, CV_8U, -1, 255);
            //      imshow("matchResult", matchResult);
            //      waitKey(0);
            //上面的直线匹配图像比原边缘图像略小，相减之前需要取得原边缘图像的子区域
            Mat barContoursNewRoi = barContoursNew(
                    Rect(lineSELen / 2 + 1, 0, videoInfo.scorebarRegion.width - lineSELen + 1, videoInfo.scorebarRegion.height));
            //      imshow("barContoursNewRoi", barContoursNewRoi);
            //      waitKey(0);

            // Mat test(barHeight, barWidth-lineSELen+1, CV_8U,Scalar(255));

            // barContoursNewRoi.convertTo(barContoursNewRoi,-1,-1,255);
            barContoursNewRoi = barContoursNewRoi - matchResult;
            //      imshow("Contours", barContoursNew);
            //      waitKey(0);
            //            Mat barContoursNewEroded(barSize.height, barSize.width,
            //            CV_8U);
            //            erode(barContoursNew, barContoursNewEroded, lineSE);
            //            imshow("barContoursNewEroded", barContoursNewEroded);
            //            waitKey(0);
            //            barContoursNew = barContoursNew-barContoursNewEroded;
            //            imshow("barContoursNew", barContoursNew);
            //            waitKey(0);

            //用Sobel算子进行梯度强度检测
            Mat sobelX(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_16S);
            Mat sobelY(videoInfo.scorebarRegion.height, videoInfo.scorebarRegion.width, CV_16S);
            Sobel(barContoursNew, sobelX, CV_16S, 1, 0, 3);
            Sobel(barContoursNew, sobelY, CV_16S, 0, 1, 3);
            Mat sobel1 = abs(sobelX) + abs(sobelY);
            double sobmax;
            minMaxLoc(sobel1, 0, &sobmax);
            sobel1.convertTo(sobel1, CV_8U, 255.0 / sobmax, 0);

            //      imshow("Contours", sobel1);
            //      waitKey(0);

            //统计边缘梯度图像在垂直方向上的投影值
            vector<int> verticalSums((unsigned long) videoInfo.scorebarRegion.width, 0);
            int sum = 0;
            for (int i = 0; i < videoInfo.scorebarRegion.height; ++i) {
                uchar *aRow = sobel1.ptr<uchar>(i);
                for (int j = 0; j < videoInfo.scorebarRegion.width; ++j) {
                    //                    if (row[j]) {
                    //                        ++x[i];
                    //                        ++sum;
                    //                    }
                    verticalSums[j] += aRow[j];
                    sum += aRow[j];
                }
            }
            cout << "Frame " << currentFrameNum << " vertical sums:" << endl;
            for (int value : verticalSums) {
                cout << value << endl;
            }
            //均值
            int verticalSumsAvg = sum / videoInfo.scorebarRegion.width;
            cout << "average: " << verticalSumsAvg << endl;
            //将小于阈值的值置为0
            int cutThreshold = 2 * 255;
            for (int &value : verticalSums) {
                if (value < cutThreshold) {
                    value = 0;
                }
            }
            cout << "Frame " << currentFrameNum << " modified vertical sums:" << endl;
            for (int value : verticalSums) {
                cout << value << endl;
            }

            //开始垂直分割图像
            int gapThreshold = (int) (videoInfo.scorebarRegion.height * 0.1);       //文字间隔阈值
            int textWidthThreshold = (int) (videoInfo.scorebarRegion.height * 0.3); //文字宽度阈值
            vector<bool> textIndicator((unsigned long) videoInfo.scorebarRegion.width, false);
            for (int i = 0; i < videoInfo.scorebarRegion.width - textWidthThreshold + 1; ++i) {
                int zeroCount = 0;
                for (int j = i; j < i + textWidthThreshold; ++j) {
                    if (verticalSums[j] == 0) {
                        ++zeroCount;
                    }
                }
                if (zeroCount <= gapThreshold) {
                    for (int k = i + 1; k < i + textWidthThreshold - 1; ++k) {
                        textIndicator[k] = true;
                    }
                }
            }

            //填补中间的总局数左右括号与局数之间可能产生的大间距
            //将填补区域限定在 中点-barHeight/2~中点+barHeight/2
            //      int frameGapFillThreshold=gapThreshold*2;
            //      int frameWidth=round(barWidth/barHeight*barHeight*1);
            //      int frameGapFillEnd=barWidth/2+frameWidth/2-frameGapFillThreshold;
            //      for (int i=barWidth/2-frameWidth/2; i<frameGapFillEnd; ++i) {
            //        if (textIndicator[i]&&textIndicator[i+frameGapFillThreshold-1])
            //        {
            //          for (int j=i+1; j<i+frameGapFillThreshold-1; ++j) {
            //            textIndicator[j]=true;
            //          }
            //        }
            //      }

            //将中间1个barHeight宽度的地方填充，确保总局分不间断
            //      int fillMaxFramesGapBegin=barWidth/2-round(barHeight*0.4);
            //      int fillMaxFramesGapEnd=barWidth/2+round(barHeight*0.4);
            //      for (int i=fillMaxFramesGapBegin; i<=fillMaxFramesGapEnd; ++i) {
            //        textIndicator[i]=true;
            //      }

            //为防止球员姓名之间的间距过大导致姓名断开，将比分条左右半边大于一定阈值的空隙进行填补
            //将填补区域限定在左边1/4和右边1/4
            int nameGapFillThreshold = gapThreshold * 3;
            int leftEnd = videoInfo.scorebarRegion.width / 4 - nameGapFillThreshold;
            int rightEnd = videoInfo.scorebarRegion.width - nameGapFillThreshold;
            for (int i = 0; i < leftEnd; ++i) {
                if (textIndicator[i] && textIndicator[i + nameGapFillThreshold - 1]) {
                    for (int j = i + 1; j < i + nameGapFillThreshold - 1; ++j) {
                        textIndicator[j] = true;
                    }
                }
            }
            for (int i = videoInfo.scorebarRegion.width * 3 / 4; i < rightEnd; ++i) {
                if (textIndicator[i] && textIndicator[i + nameGapFillThreshold - 1]) {
                    for (int j = i + 1; j < i + nameGapFillThreshold - 1; ++j) {
                        textIndicator[j] = true;
                    }
                }
            }

            //测试结果
            cout << "Frame " << currentFrameNum << " text areas:" << endl;
            for (bool value : textIndicator) {
                cout << (value ? "#####" : "0") << endl;
            }

            //确定各个字段的位置
            int maxFrameNumPos[2] = {0, 0};
            int frameNum1Pos[2] = {0, 0};
            int frameNum2Pos[2] = {0, 0};
            int score1Pos[2] = {0, 0};
            int score2Pos[2] = {0, 0};
            int name1Pos[2] = {0, 0};
            int name2Pos[2] = {0, 0};
            int name1PosPlanB[2] = {0, 0};
            int name2PosPlanB[2] = {0, 0};

            //向左检测
            int cnt = 0;
            for (int i = videoInfo.scorebarRegion.width / 2; i >= 0; --i) {
                if (textIndicator[i] == false && textIndicator[i + 1] == true) {
                    if (cnt == 0) {
                        maxFrameNumPos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 2) {
                        frameNum1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 4) {
                        score1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 6) {
                        name1Pos[0] = i + 1;
                        ++cnt;
                    }
                    if (cnt == 8) {
                        name1PosPlanB[0] = i + 1;
                        break;
                    }
                }
                if (textIndicator[i] == true && textIndicator[i + 1] == false) {
                    if (cnt == 1) {
                        frameNum1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 3) {
                        score1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 5) {
                        name1Pos[1] = i;
                        ++cnt;
                    }
                    if (cnt == 7) {
                        name1PosPlanB[1] = i;
                        ++cnt;
                    }
                }
            }
            //向右检测
            cnt = 0;
            for (int i = videoInfo.scorebarRegion.width / 2; i < videoInfo.scorebarRegion.width; ++i) {
                if (textIndicator[i] == false && textIndicator[i - 1] == true) {
                    if (cnt == 0) {
                        maxFrameNumPos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 2) {
                        frameNum2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 4) {
                        score2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 6) {
                        name2Pos[1] = i - 1;
                        ++cnt;
                    }
                    if (cnt == 8) {
                        name2PosPlanB[1] = i - 1;
                        break;
                    }
                }
                if (textIndicator[i] == true && textIndicator[i - 1] == false) {
                    if (cnt == 1) {
                        frameNum2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 3) {
                        score2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 5) {
                        name2Pos[0] = i;
                        ++cnt;
                    }
                    if (cnt == 7) {
                        name2PosPlanB[0] = i;
                        ++cnt;
                    }
                }
            }

            //如果两个名字区域的宽度都小于一个阈值，则启用namePosPlanB
            int nameWidthThreshold = (int) round(videoInfo.scorebarRegion.height * 1.2);
            if (name1Pos[1] - name1Pos[0] < nameWidthThreshold && name2Pos[1] - name2Pos[0] < nameWidthThreshold) {
                name1Pos[0] = name1PosPlanB[0];
                name1Pos[1] = name1PosPlanB[1];
                name2Pos[0] = name2PosPlanB[0];
                name2Pos[1] = name2PosPlanB[1];
            }

            //在原图上画线标示识别出的文字区域
//            rectangle(scorebar1, Point(maxFrameNumPos[0], 0),
//                    Point(maxFrameNumPos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(frameNum1Pos[0], 0),
//                    Point(frameNum1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(frameNum2Pos[0], 0),
//                    Point(frameNum2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(score1Pos[0], 0),
//                    Point(score1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(score2Pos[0], 0),
//                    Point(score2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(name1Pos[0], 0),
//                    Point(name1Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
//            rectangle(scorebar1, Point(name2Pos[0], 0),
//                    Point(name2Pos[1], barHeight - 1), Scalar(0, 255, 0), 1);
#ifdef DEBUG_SCORE_BAR_DETECTION
            imshow("Score Bar", scorebar1);
            waitKey(0);
#endif
            //如果这些区域都存在的话, 则保存此时的比分条区域
            if (maxFrameNumPos[0] && maxFrameNumPos[1]
                    && frameNum1Pos[0] && frameNum1Pos[1]
                    && frameNum2Pos[0] && frameNum2Pos[1]
                    && score1Pos[0] && score1Pos[1]
                    && score2Pos[0] && score2Pos[1]
                    && name1Pos[0] && name1Pos[1]
                    && name2Pos[0] && name2Pos[1]) {
                frameFeature.maxFrameNumPos[0] = maxFrameNumPos[0];
                frameFeature.maxFrameNumPos[1] = maxFrameNumPos[1];
                frameFeature.frameNum1Pos[0] = frameNum1Pos[0];
                frameFeature.frameNum1Pos[1] = frameNum1Pos[1];
                frameFeature.frameNum2Pos[0] = frameNum2Pos[0];
                frameFeature.frameNum2Pos[1] = frameNum2Pos[1];
                frameFeature.score1Pos[0] = score1Pos[0];
                frameFeature.score1Pos[1] = score1Pos[1];
                frameFeature.score2Pos[0] = score2Pos[0];
                frameFeature.score2Pos[1] = score2Pos[1];
                frameFeature.name1Pos[0] = name1Pos[0];
                frameFeature.name1Pos[1] = name1Pos[1];
                frameFeature.name2Pos[0] = name2Pos[0];
                frameFeature.name2Pos[1] = name2Pos[1];
                //frameFeature.scorebarRegion = tmpScorebarRegion;
                frameFeature.hasScoreBar = true;
            }
        }
        //OCR之后写入frameFeature中
    } //if (isFullTableView)
}

void SnookerVideoEventDetector::GetScorebarRegion() {
    FrameFeature frameFeature;
    //获取一定数量的候选比分条区域, 再从中确定最终的比分条区域
    const int candicateScorebarRegionNum = 50;
    int fps = static_cast<int>(videoInfo.fps);
    int candicates = 0;
    for (unsigned int currentFrameNum = 100;
         currentFrameNum < videoInfo.framesNum && candicates < candicateScorebarRegionNum;
         currentFrameNum += fps /*每隔一秒取一个样本帧*/, ++candicates) {
        //只对不属于回放镜头的帧进行处理
        if (videoInfo.replayCheckHashTab.find(currentFrameNum) == videoInfo.replayCheckHashTab.end()) {
            GetFrameFeature(currentFrameNum, frameFeature);
            //如果这些区域都存在的话, 则保存此时的比分条区域
            if (frameFeature.maxFrameNumPos[0] && frameFeature.maxFrameNumPos[1] && frameFeature.frameNum1Pos[0] &&
                    frameFeature.frameNum1Pos[1] && frameFeature.frameNum2Pos[0] && frameFeature.frameNum2Pos[1] &&
                    frameFeature.score1Pos[0] && frameFeature.score1Pos[1] && frameFeature.score2Pos[0] &&
                    frameFeature.score2Pos[1] && frameFeature.name1Pos[0] && frameFeature.name2Pos[1]) {
                candidateScorebarRegions.push_back(frameFeature.scorebarRegion);
            }
        }
    } // for每一个样本帧

    //从候选比分条区域中确定最终的比分条区域
    int finalX, finalY, finalHeight;
    vector<int> vecX, vecY, vecHeight;
    vector<Rect>::const_iterator it = candidateScorebarRegions.begin();
    for (; it != candidateScorebarRegions.end(); ++it) {
        vecX.push_back((*it).x);
        vecY.push_back((*it).y);
        vecHeight.push_back((*it).height);
    }
    sort(vecX.begin(), vecX.end());
    sort(vecY.begin(), vecY.end());
    sort(vecHeight.begin(), vecHeight.end());

    finalX = vecX[vecX.size() / 2];
    finalY = vecY[vecY.size() / 2];
    finalHeight = vecHeight[vecHeight.size() / 2];

    //将最终的比分条位置写入videoInfo中
    videoInfo.scorebarRegion = Rect(finalX, finalY, videoInfo.width - 2 * finalX, finalHeight);
}

void SnookerVideoEventDetector::GetGrayScorebarFromFrameId(int const frameId, Mat &scorebar) {
    //读取该帧图像
    string imgPath(videoInfo.framesFolder);
    imgPath += "frame" + to_string(frameId) + ".jpg";
    Mat srcImg = imread(imgPath);
    scorebar = srcImg(videoInfo.scorebarRegion).clone();

    //将比分条区域灰度化
    cvtColor(scorebar, scorebar, CV_BGR2GRAY, 1);
#ifdef DEBUG_TURN_INDICATOR
    imshow("Turn indicator", scorebar);
    waitKey(0);
#endif
}

bool SnookerVideoEventDetector::GetCurrentPlayerFlagPos() {
    int fps = static_cast<int>(videoInfo.fps);
    for (unsigned int currentFrameNum = 100;
         currentFrameNum < videoInfo.framesNum;
         currentFrameNum += fps /*每隔一秒取一个样本帧*/) {
        //只对不属于回放镜头的帧进行处理
        if (videoInfo.replayCheckHashTab.find(currentFrameNum) == videoInfo.replayCheckHashTab.end()) {
            FrameFeature frameFeature;
            GetFrameFeature(currentFrameNum, frameFeature);
            //如果这些区域都存在的话, 则保存此时的比分条区域
            if (frameFeature.hasScoreBar) {
                if (lastScorebar.data == NULL) {
                    GetGrayScorebarFromFrameId(currentFrameNum, lastScorebar);
                    continue;
                }
                else {
                    GetGrayScorebarFromFrameId(currentFrameNum, currentScorebar);
                    //计算两帧之差
                    Mat scorebarDiff = currentScorebar - lastScorebar;

                    //测试
                    imshow("Scorebar diff", scorebarDiff);
                    waitKey(0);

                    if (DetectTurnIndicator(scorebarDiff)) {
                        return true;
                    }
                    else {
                        lastScorebar = currentScorebar;
                        continue;
                    }
                }
            }
        }
    } // for每一个样本帧
    return false;
}

void SnookerVideoEventDetector::GetVideoFramesFeature() {
    int fps = static_cast<int>(videoInfo.fps);
    for (unsigned int currentFrameNum = 0; currentFrameNum < videoInfo.framesNum; currentFrameNum += fps/*每隔一秒取一个样本帧*/) {
        //只对不属于回放镜头的帧进行处理
        if (videoInfo.replayCheckHashTab.find(currentFrameNum) == videoInfo.replayCheckHashTab.end()) {
            FrameFeature frameFeature;
            GetFrameFeature(currentFrameNum, frameFeature);
        }//if (videoInfo.replayCheckHashTab.find(currentFrameNum)==videoInfo.replayCheckHashTab.end())

    }//for每一个样本帧
}

//将直线段集中的直线段画到图像中，便于观察
void SnookerVideoEventDetector::DrawDetectedLinesP(Mat &image, const vector<Vec4i> &lines, Scalar &color) {
    // 将检测到的直线在图上画出来
    vector<Vec4i>::const_iterator it = lines.begin();
    while (it != lines.end()) {
        Point pt1((*it)[0], (*it)[1]);
        Point pt2((*it)[2], (*it)[3]);
        line(image, pt1, pt2, color, 1); //  线条宽度设置为2
        ++it;
    }
}

//将直线集中的直线画到图像中，便于观察
void SnookerVideoEventDetector::DrawDetectedLines(Mat &image, const vector<Vec2f> &lines,
        const Scalar &color, const int width) {
    vector<Vec2f>::const_iterator lineIt = lines.begin();
    while (lineIt != lines.end()) {
        float rho = (*lineIt)[0];
        float theta = (*lineIt)[1];
        if (theta < CV_PI / 4.0 || theta > 3.0 * CV_PI / 4.0) {
            Point pt1(rho / cos(theta), 0);
            Point pt2((rho - image.rows * sin(theta)) / cos(theta), image.rows);
            line(image, pt1, pt2, color, width);
            //            imshow("Image", image);
            //            waitKey(0);
        } else {
            Point pt1(0, rho / sin(theta));
            Point pt2(image.cols, (rho - image.cols * cos(theta)) / sin(theta));
            line(image, pt1, pt2, color, width);
            //            imshow("Image", image);
            //            waitKey(0);
        }
        ++lineIt;
    }
}

//计算两条直线的交点，将交点加入到点集points中
void SnookerVideoEventDetector::IntersectionPoint(const Vec2f line1, const Vec2f line2,
        vector<Point> &points, int imgRows, int imgCols) {
    double a11 = cos(line1[1]), a12 = sin(line1[1]), b1 = line1[0];
    double a21 = cos(line2[1]), a22 = sin(line2[1]), b2 = line2[0];
    double d = a11 * a22 - a12 * a21;

    //判断系数行列式是否为零
    if (d > -0.0000001 && d < 0.0000001) {
        return;
    } else {
        double d1 = b1 * a22 - a12 * b2, d2 = a11 * b2 - b1 * a21;
        int x = static_cast<int>(round(d1 / d)),
                y = static_cast<int>(round(d2 / d));

        //判断x与y是否在图像范围之内
        if (x < 0 || x >= imgCols || y < 0 || y >= imgRows)
            return;
        points.push_back(Point(x, y));
    }
    return;
}

//在指定的位置画圆圈
void SnookerVideoEventDetector::DrawCircles(Mat &image, const vector<Point> &points, const Scalar &color) {
    vector<Point>::const_iterator it = points.begin();
    for (; it != points.end(); ++it) {
        circle(image, *it, 7, color, 2);
    }
}

//在交点集合中选取离左右两边缘最近的一个点，之后以此点为基准截取比分条区域
void SnookerVideoEventDetector::SelectPoint(const vector<Point> &points, Size imageSize,
        Point &selectedPoint, bool &leftSide) {
    int dist = 99999;
    Point myPoint;
    vector<Point>::const_iterator ptIt = points.begin();
    for (; ptIt != points.end(); ++ptIt) {
        int x = (*ptIt).x;
        if (x < imageSize.width / 2) {
            int currDist = x;
            if (currDist < dist) {
                myPoint = *ptIt;
                leftSide = true;
                dist = currDist;
            }
        } else {
            int currDist = imageSize.width - x;
            if (currDist < dist) {
                myPoint = *ptIt;
                leftSide = false;
                dist = currDist;
            }
        }
    }
    selectedPoint = myPoint;
}

//计算桌面区域的高度
int SnookerVideoEventDetector::TableRegionHeight(const vector<Point> &points, const int imageHeight) {
    int size = static_cast<int>(points.size());
    if (size < 2)
        return -1;
    int max = 0, min = 99999;
    for (int i = 0; i < size; ++i) {
        int y = points[i].y;
        if (y > max)
            max = y;
        if (y < min)
            min = y;
    }
    int tableRegionHeight = max - min;
    //如果检测出来的桌面区域高度小于图像高度的1/3，显然不正确，返回-1
    if (tableRegionHeight < imageHeight / 3)
        return -1;
    return tableRegionHeight;
}

//位置相近的直线仅保留一条，移除多余的直线
void SnookerVideoEventDetector::RemoveNearbyLines(vector<Vec2f> &lines, const Size imageSize) {
    if (lines.size() < 2)
        return;
    float rhoThreshold = imageSize.height * 0.2;
    float thetaThreshold = CV_PI / 180 * 5;

    vector<Vec2f> consideredLines;
    consideredLines.push_back(lines[0]);
    for (int i = 1; i != lines.size(); ++i) {
        bool reserve = true;
        for (int j = 0; j != consideredLines.size(); ++j) {
            if (abs(lines[i][0] - consideredLines[j][0]) < rhoThreshold &&
                    abs(lines[i][1] - consideredLines[j][1]) < thetaThreshold) {
                reserve = false;
                break;
            }
        }
        if (reserve)
            consideredLines.push_back(lines[i]);
    }
    lines = consideredLines;
    return;
}

//位置相近的点只保留一个，移除多余的点
void SnookerVideoEventDetector::RemoveNearbyPoints(vector<Point> &points, const Size imageSize) {
    if (points.size() < 2)
        return;
    float thresholdRatio = 0.4; //点之间的间隔阈值比例
    float pointsDistantThreshold = imageSize.height * thresholdRatio; //实际阈值
    vector<Point> consideredPoints;
    consideredPoints.push_back(points[0]);
    for (int i = 1; i != points.size(); ++i) {
        bool reserve = true;
        for (int j = 0; j != consideredPoints.size(); ++j) {
            if (abs(points[i].x - consideredPoints[j].x) < pointsDistantThreshold &&
                    abs(points[i].y - consideredPoints[j].y) < pointsDistantThreshold) {
                reserve = false;
                break;
            }
        }
        if (reserve)
            consideredPoints.push_back(points[i]);
    }
    points = consideredPoints;
    return;
}

//根据直线的角度和位置判定图像是否为全台面视角
bool SnookerVideoEventDetector::IsFullTableView(const vector<Vec2f> &lines, const Size imageSize) {
    if (lines.size() != 4) {
        return false;
    }
    float upLow = 0.03, upHigh = 0.3, downLow = 0.6, downHigh = 0.9,
            leftLow = 0.1, leftHigh = 0.4, rightLow = -0.9, rightHigh = -0.5;
    bool up = false, down = false, left = false, right = false;
    for (int i = 0; i < 4; ++i) {
        if (lines[i][1] > CV_PI / 2 - CV_PI / 180 * 3 &&
                lines[i][1] < CV_PI / 2 + CV_PI / 180 * 3) { //两条水平线之一
            if (lines[i][0] > imageSize.height * upLow &&
                    lines[i][0] < imageSize.height * upHigh) {
                up = true;
            } else if (lines[i][0] > imageSize.height * downLow &&
                    lines[i][0] < imageSize.height * downHigh) {
                down = true;
            }
        } else if (lines[i][1] > CV_PI / 180 * 5 &&
                lines[i][1] < CV_PI / 180 * 25) { //左边的斜线
            if (lines[i][0] > imageSize.width * leftLow &&
                    lines[i][0] < imageSize.width * leftHigh) {
                left = true;
            }
        } else if (lines[i][1] > CV_PI / 180 * 155 &&
                lines[i][1] < CV_PI / 180 * 175) { //右边的斜线
            cout << lines[i][0] << endl;
            if (lines[i][0] > imageSize.width * rightLow &&
                    lines[i][0] < imageSize.width * rightHigh) {
                right = true;
            }
        }
    }
    return up && down && left && right;
}

//通过膨胀再腐蚀(闭操作)移除图像中的小物体，比如会影响直线检测的球杆、手臂等
void SnookerVideoEventDetector::RemoveSmallObjects(Mat &image, const Size &imageSize) {
    int seSize = imageSize.height * 0.02;
    Mat se(seSize, seSize, CV_8U, Scalar(1));
    dilate(image, image, se);
    erode(image, image, se);
}

// bool IsFullTableView(const vector<Point>& points, const Size imageSize,
// float& value)
//{
//    if (points.size() < 4) {
//        value = 0;
//        return false;
//    }
//
//    //将点的坐标归一化
//    vector<Point2f> pointsNorm;
//    for (int i = 0; i < points.size(); ++i) {
//        pointsNorm.push_back(Point2f(points[i].x * 1.0 / imageSize.width,
//        points[i].y * 1.0 / imageSize.height));
//    }
//
//    if (pointsNorm.size() < 4) {
//        value = 0;
//        return false;
//    }
//    value = 0;
//    for (int i = 0; i < pointsNorm.size(); ++i) {
//        value += pow(pointsNorm[i].x - 0.5, 2) + pow(pointsNorm[i].y - 0.5,
//        2);
//    }
//
//    return true;
//}

/**
*  测试函数, 用来测试直方图的显示, 暂时没使用
*
*  @param imagePath <#imagePath description#>
*
*  @return <#return value description#>
*/
int ShowHistogram(const char *imagePath) {
    Mat src, hsv;
    if (!(src = imread(imagePath, 1)).data)
        return -1;

    cvtColor(src, hsv, CV_BGR2HSV);

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = {0, 180};
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float *ranges[] = {hranges, sranges};
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist(&hsv, 1, channels, Mat(), // do not use mask
            hist, 2, histSize, ranges,
            true, // the histogram is uniform
            false);
    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scale = 10;
    Mat histImg = Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

    for (int h = 0; h < hbins; h++)
        for (int s = 0; s < sbins; s++) {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(histImg, Point(h * scale, s * scale),
                    Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                    Scalar::all(intensity), CV_FILLED);
        }

    namedWindow("Source", 1);
    imshow("Source", src);

    namedWindow("H-S Histogram", 1);
    imshow("H-S Histogram", histImg);
    waitKey();
    return 0;
}

SnookerVideoEventDetector::SnookerVideoEventDetector() {
    videoInfo.fps = 0;                        //帧率
    videoInfo.framesNum = 0;                  //总帧数
    videoInfo.width = 0;                      //视频宽度
    videoInfo.height = 0;                     //视频高度
    videoInfo.bestFrames = 0;                 //总局数

    //比分条在原帧中的位置
    videoInfo.scorebarRegion.x = 0;
    videoInfo.scorebarRegion.y = 0;
    videoInfo.scorebarRegion.width = 0;
    videoInfo.scorebarRegion.height = 0;

    //当前击球球员指示符在比分条中的位置
    videoInfo.currentPlayerFlagPos[0][0] = 0;
    videoInfo.currentPlayerFlagPos[0][1] = 0;
    videoInfo.currentPlayerFlagPos[1][0] = 0;
    videoInfo.currentPlayerFlagPos[1][1] = 0;
}

bool SnookerVideoEventDetector::DetectTurnIndicator(cv::Mat &scorebarDiff) {
    //将比分区域之外的部分向垂直方向投影
    float ratio = 0.35; //左右两侧比分区域的比例
    vector<int> flags(videoInfo.scorebarRegion.width, 0);
    for (int i = 0; i < videoInfo.scorebarRegion.height; ++i) {
        uchar *row = scorebarDiff.ptr<uchar>(i);
        int leftEndPos = static_cast<int>(videoInfo.scorebarRegion.width * ratio);
        for (int j = 0; j < leftEndPos; ++j) {
            if (row[j] > 60) {
                flags[j] += 1;
            }
        }
        int rightStart = static_cast<int>(videoInfo.scorebarRegion.width * (1 - ratio));
        for (int j = rightStart; j < videoInfo.scorebarRegion.width; ++j) {
            if (row[j] > 60) {
                flags[j] += 1;
            }
        }
    }

    //判断数组 flags 中是否含有合法的指示标志
    //从前往后找到第一个非零位置，从后往前找到第一个非零位置
    int idx1 = 0, idx2 = videoInfo.scorebarRegion.width - 1;
    while (!flags[idx1] && idx1 <= videoInfo.scorebarRegion.width) {
        ++idx1;
    }
    if (idx1 == videoInfo.scorebarRegion.width) {  //全部为0，不可能有指示符
        return false;
    }
    //找到结束位置
    while (!flags[idx2] && idx2 > idx1) {
        --idx2;
    }
    int cnt = 0;
    for (int i = idx1; i <= idx2; ++i) {
        ++cnt;
        if (!flags[i]) {
            return false; //中间有间隙，不是指示符
        }
    }
    //判断宽度
    if (cnt > videoInfo.scorebarRegion.height * 0.1 && cnt < videoInfo.scorebarRegion.height * 0.6) {
        //判断是左边还是右边
        if (idx1 < videoInfo.scorebarRegion.width / 2) {   //左边
            videoInfo.currentPlayerFlagPos[0][0] = idx1;
            videoInfo.currentPlayerFlagPos[0][1] = idx2;
            videoInfo.currentPlayerFlagPos[1][0] = videoInfo.scorebarRegion.width - 1 - idx2;
            videoInfo.currentPlayerFlagPos[1][1] = videoInfo.scorebarRegion.width - 1 - idx1;
        }
        else {  //右边
            videoInfo.currentPlayerFlagPos[0][0] = videoInfo.scorebarRegion.width - 1 - idx2;
            videoInfo.currentPlayerFlagPos[0][1] = videoInfo.scorebarRegion.width - 1 - idx1;
            videoInfo.currentPlayerFlagPos[1][0] = idx1;
            videoInfo.currentPlayerFlagPos[1][1] = idx2;
        }
        return true;
    }
    else {
        return false;
    }
}
