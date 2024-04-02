import multiprocessing
import sys
import os
import cv2 as cv
import numpy as np
import cv2

def longTimeBinary(src):
    adth = np.zeros(src.shape, np.uint16)
    bian = 2 ** 4
    for i in range(0, 2 ** 9 - bian):
        for j in range(0, 2 ** 9 - bian):
            tmpImg = src[i:i + bian, j:j + bian]
            ret, th = cv2.threshold(tmpImg, tmpImg.mean(), 2 ** 16 - 1, cv2.THRESH_BINARY)
            th[np.where(th != 2 ** 16 - 1)] = 0
            th[np.where(th == 2 ** 16 - 1)] = 1
            adth[i:i + bian, j:j + bian] += th

    adth[np.where(adth > 128)] = 2 ** 16 - 1  # 512=((2**5)^2)/2
    adth[np.where(adth <= 128)] = 0

    return adth



def processPoint(path):
    print("processing " + path)
    fitcPath = path
    fitcBinPath = path + "/binaryG/"
    try:
        os.mkdir(fitcBinPath)
    except Exception as e:
        print(e)


    fitcNames = os.listdir(fitcPath)
    for name in fitcNames:
        if ".tif" in name:
            print(fitcPath + name)
            tmpImg = cv.imread(fitcPath + name, -1)
            tmpImg = longTimeBinary(tmpImg)
            #cv.imwrite(fitcBinPath + list(name.split("-"))[0] + ".tif", tmpImg)
            cv.imwrite(fitcBinPath + str(int(name.split('.')[0])) + ".tif", tmpImg)

def processThreading(pathList):
    print("In longTimeBinary.py thread, Processing:")
    print("<_____---^_^---_____>")
    print(pathList)
    print("<_____---^_^---_____>")
    for path in pathList:
        processPoint(path)


if __name__ == "__main__":
    coreNum = int(sys.argv[1])
    pointSetsPath = "D:/li/pointSets/20240229/1/"
    allPathNames = os.listdir(pointSetsPath)
    pointPathNames = []
    for d in allPathNames:
        if "." not in d:
            pointPathNames.append(pointSetsPath + d + "/")
    step = int(len(pointPathNames)/coreNum) + 1
    pathsForth = []
    divNum = 0
    while divNum * step < len(pointPathNames):
        pathsForth.append(pointPathNames[divNum * step:min(len(pointPathNames), (divNum + 1) * step)])
        divNum += 1
    threads = []
    for paths in pathsForth:
        tmpTh = multiprocessing.Process(target=processThreading, args=(paths,))
        threads.append(tmpTh)
    for th in threads:
        th.start()
