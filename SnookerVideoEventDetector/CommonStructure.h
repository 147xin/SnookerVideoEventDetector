#pragma once

#define HISTBINS    16
//直方图定义
typedef struct HistData {
    int h[HISTBINS];
    int s[HISTBINS];
    int v[HISTBINS];
} HistData, *pHistData;

//渐变信息
typedef struct _GTInfo {
    int nStart;
    //开始位置
    int nLength;//渐变长度
} GTInfo;

//升序排列
extern bool less_start(const GTInfo &info1, const GTInfo &info2);

//回放镜头
typedef struct _ReplayInfo {
    GTInfo start;
    GTInfo end;
} ReplayInfo;
