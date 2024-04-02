from tempfile import gettempdir
from xml.dom import minidom
import cv2 as cv
import numpy as np
from steper import steper
import os
from pycromanager import Bridge
#读取目标图片
#target = cv.imread("target.png")
#读取模板图片
#template = cv.imread("template.png")
#获得模板图片的高宽尺寸
#theight, twidth = template.shape[:2]
#执行模板匹配，采用的匹配方式cv.TM_SQDIFF_NORMED
#result = cv.matchTemplate(target,template,cv.TM_SQDIFF_NORMED)
#归一化处理
#cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
#寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
#匹配值转换为字符串
#对于cv.TM_SQDIFF及cv.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
#对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
#strmin_val = str(min_val)
#绘制矩形边框，将匹配区域标注出来
#min_loc：矩形定点
#(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
#(0,0,225)：矩形的边框颜色；2：矩形边框宽度
#cv.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
#显示结果,并将匹配值显示在标题栏上
#cv.imshow("MatchResult----MatchingValue="+strmin_val,target)
#cv.waitKey()
#cv.destroyAllWindows()

def getTemplateLoc(template, target):
    result = cv.matchTemplate(target, template, cv.TM_CCORR_NORMED)
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return (max_loc, max_val)

def getSnap():
    live = studio.live()
    live.snap(True)
    taggedImaged = core.get_tagged_image()
    pixels = np.reshape(
        taggedImaged.pix,
        newshape=[taggedImaged.tags['Height'], taggedImaged.tags['Width']]
        )
    return pixels

def uint16To8ThenOtsu(src):
    tmp = np.zeros(src.shape, dtype=np.uint8)
    src = src/np.max(src)*(2**8-1)
    tmp[:,:] = src.astype(np.uint8)
    # ret, binary = cv.threshold(tmp, 0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    return tmp

def multiPointMatch(template, points, delta, target):
    locValList = []
    for p in points:
        h = p[0]
        w = p[1]
        loc, val = getTemplateLoc(template[h-delta:h+delta, w-delta:w+delta], target)
        locValList.append([loc,val])
    locValList = np.array(locValList)
    print(locValList)
    minIndex = np.where(locValList[:,1]==np.min(locValList[:,1]))[0][0]
    print(minIndex)
    ox, oy = points[minIndex]
    cx, cy = locValList[minIndex, 0]
    return [cy - oy + delta, cx - ox + delta]
    


def main():
    mySteper = steper()
    while True:
        cmd = input()
        if cmd == "2":
            img = getSnap()
            cv.imwrite(DIR + "target.tif", img)
        elif cmd == "1":
            template = getSnap()
            target = cv.imread(DIR + "target.tif", -1)
            template = uint16To8ThenOtsu(template)
            target = uint16To8ThenOtsu(target)
            points = [
                [1000, 1000],
            ]
            dx,dy = multiPointMatch(template, points, 500, target)
            print(dx, dy)
            mySteper.xMoveRelative(-dx * 0.065)
            mySteper.yMoveRelative(dy * 0.065)
            

if __name__ == "__main__":
    DIR = os.path.dirname(__file__) + "\\"
    bridge = Bridge()
    core = bridge.get_core()
    studio = bridge.get_studio()
    main()
    # a = np.array([
    #     [1,2,3],
    #     [2,3,4],
    #     [3,4,5]
    # ], dtype=np.uint8)
    # b = np.array([
    #     [3,4],
    #     [4,5]
    # ], dtype=np.uint8)
    # print(getTemplateLoc(b,a))