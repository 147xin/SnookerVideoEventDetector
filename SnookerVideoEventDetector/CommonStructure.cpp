//#include "StdAfx.h"
#include "CommonStructure.h"

//按值升序排列
bool less_start(const GTInfo &info1, const GTInfo &info2) {
    return info1.nStart < info2.nStart;
}
