import traceback
import csv
import os
import sys
from ctypes import util
from itertools import count
from platform import release
from re import template
from subprocess import TimeoutExpired
from unicodedata import category
from scipy import stats
import cv2
from delta import utilities as utils
import numpy as np
import pickle
import matplotlib.pyplot as plt
import multiprocessing


def getInternal(contours, cells, height, width):
    reDict = {}
    for i in range(len(cells)):
        tmpA = np.zeros((height, width), np.uint8)
        tmpA = cv2.drawContours(
            tmpA,
            contours,
            i,
            color=1,
            thickness=-1,
            offset=(0, 0)
        )
        reDict[cells[i]] = (np.array(np.where(tmpA == 1)).T)
    return reDict


def getDivisionLengths(cellDict):
    divisionLengths = []
    for p in cellDict["0"]:
        lengths = p["length"]
        daughters = p["daughters"]
        for i in range(len(lengths)):
            if daughters[i] != None:
                divisionLengths.append(lengths[i - 1])
    return divisionLengths


def getDivisionTimes(cellDict):
    divisionTimes = []
    for p in cellDict["0"]:
        daughters = p["daughters"]
        for i in range(len(daughters)):
            if daughters[i] != None:
                for j in range(i):
                    if daughters[i - j - 1] != None or j == i - 1:
                        divisionTimes.append(j)
                        break
    return divisionTimes


def getCellDict(pklPath):
    with open(pklPath, "rb") as f:
        pkl = pickle.load(f)
    cellDict = {}
    n = 0
    for r, roi in enumerate(pkl.rois):
        cellDict[str(n)] = roi.lineage.cells
        n += 1
    return cellDict


def getIndexInFrame(cell, frameCells):
    matchList = []
    for i in range(len(cell["frame"])):
        for c in frameCells[i]:
            if c["new_pole"][i] in cell:
                matchList.append(i)
                break
    return matchList


def getIndexList(pklPath):
    reList = []
    cells = getCellDict(pklPath)
    frameCells = getFrameCells(pklPath)
    for cell in cells:
        reList.append(getIndexInFrame(cell, frameCells))
    return reList


def getFrameCells(path):
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    masks = pkl.rois[0].label_stack
    print(path)
    frameCells = []
    for i in range(len(masks)):
        print("processing" + str(i))
        cells, contours = utils.getcellsinframe(masks[i], return_contours=True)
        frameCells.append(getInternal(contours, cells, 512, 512))
    return frameCells


def checkPercent(img, points):
    maxGray = np.max(img)
    num = 0
    for p in points:
        if len(img.shape) != 2:
            if (img[p[0], p[1]] == [255, 255, 255]).all():
                num += 1
        else:
            if img[p[0], p[1]] == maxGray:
                num += 1

    return num / len(points)


def cellCon(img, cellContours):
    fitc = (img / np.max(img) * (2 ** 9 - 1)).astype(np.uint8)
    fitcRGB = cv2.cvtColor(fitc, cv2.COLOR_GRAY2RGB)
    fitcRGB = cv2.drawContours(
        fitcRGB,
        cellContours,
        -1,
        color=(255, 255, 0),
        thickness=1,
        offset=(0, 0)
    )
    return fitcRGB


def drawCircles(shape, radius):
    h, w = shape
    circles = []
    for x in range(radius, w - radius):
        for y in range(radius, h - radius):
            pureCircle = cv2.circle(
                img=np.zeros(shape),
                center=(x, y),
                radius=radius,
                color=1,
                thickness=-1
            )
            circles.append(pureCircle)
    return circles


def checkCellContainCircle(cell, circle):
    maxPixelCount = np.sum(circle)
    overlapCount = len(np.where((circle + cell) == 2)[0])
    if overlapCount == maxPixelCount:
        return True
    return False


def isGreen(contour, binaryImg):
    minH, maxH = min(contour[:, 0]), max(contour[:, 0])
    minW, maxW = min(contour[:, 1]), max(contour[:, 1])
    binaryCell = binaryImg[minH:maxH, minW:maxW]
    binaryCell[np.where(binaryCell != 0)] = 1
    justContour = np.zeros(binaryCell.shape)
    for hw in contour:
        h, w = hw
        justContour[h - minH - 1, w - minW - 1] = 1
    circles = drawCircles(binaryCell.shape, 3)
    for i, circle in enumerate(circles):
        if checkCellContainCircle(binaryCell, circle):
            if checkCellContainCircle(justContour, circle):
                return True
    return False


def identify(binaryPath, internalDicts):
    ifDicts = []
    for i in range(len(internalDicts)):
        greenOrRed = {"g": [], "r": []}
        img = cv2.imread(binaryPath + str(i) + ".tif", -1)
        for key in internalDicts[i]:
            if isGreen(internalDicts[i][key], img) or checkPercent(img, internalDicts[i][key]) > 0.8:
                greenOrRed["g"].append(key)
            else:
                greenOrRed["r"].append(key)
        ifDicts.append(greenOrRed)
    return (ifDicts)


def writeColorCsv(path, colorList, cellNum, frameNum):
    DIR = path
    with open(DIR + "color.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + ["frame-" + str(x) for x in range(frameNum)])
        for i in range(cellNum):
            row = [i]
            for j in range(frameNum):
                if i in colorList[j]["g"]:
                    row.append(1)
                elif i in colorList[j]["r"]:
                    row.append(2)
                else:
                    row.append(0)
            writer.writerow(row)


def writeContactCsv(path, colorList, frameCells):
    DIR = path
    frames = len(frameCells)
    try:
        os.mkdir(DIR + "contact")
    except Exception as e:
        print(e)
    for i in range(frames):
        print("contact frame" + str(i))
        tmpG = {}
        tmpR = {}
        for g in colorList[i]["g"]:
            x = frameCells[i][g][:, 0].mean()
            y = frameCells[i][g][:, 1].mean()
            tmpG[g] = [x, y]
        for r in colorList[i]["r"]:
            x = frameCells[i][r][:, 0].mean()
            y = frameCells[i][r][:, 1].mean()
            tmpR[r] = [x, y]
        with open(DIR + "contact/frame" + str(i) + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id"])
            for keyG in tmpG:
                row = [keyG]
                for keyR in tmpR:
                    if getDistance(tmpG[keyG], tmpR[keyR]) < 100:
                        if getMinDistance(frameCells[i][keyG], frameCells[i][keyR]) < 10:
                            row.append(keyR)
                writer.writerow(row)


def getMinDistance(pointsA, pointsB):
    min = 4096
    for a in pointsA:
        for b in pointsB:
            if getDistance(a, b) < min:
                min = getDistance(a, b)
    return min


def getDistance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)


def processPoint(path, pklPath, binaryFitcPath):
    with open(pklPath, "rb") as f:
        pkl = pickle.load(f)
    cells = pkl.rois[0].lineage.cells
    frameCells = getFrameCells(pklPath)
    colorList = identify(binaryFitcPath, frameCells)
    writeColorCsv(path, colorList, len(cells), len(frameCells))
    writeContactCsv(path, colorList, frameCells)


def processThreading(pathList):
    print("In pklAnalysis.py thread, Processing:")
    print("<_____---^_^---_____>")
    print(pathList)
    print("<_____---^_^---_____>")
    for path in pathList:
        try:
            processPoint(
                path,
                path + "delta_results/Position000000.pkl",
                path + "binaryG/"
            )
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    path = "D:/li/20240229/1/"


    processPoint(
            path,
            path + "/delta_results/Position000000.pkl",
            path + "FITC/binaryG/"
    )

