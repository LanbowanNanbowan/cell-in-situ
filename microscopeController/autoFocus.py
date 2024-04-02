import csv
import imp
import os
import time
from tkinter.constants import W
from cv2 import blur, setUseOpenVX
from matplotlib.quiver import QuiverKey
from numpy.core import getlimits
import matplotlib.pyplot as plt
import win32gui
import win32api
import win32ui
import win32con
import numpy as np
import pygame
import cv2 as cv
from pycromanager import Bridge

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
        left, top, x2, y2 = region
        width = x2 - left
        height = y2 - top
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    # print(img.shape)
    img.shape = (height, width, 4)
    # print(img.shape)
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    img = cv.cvtColor(img[:,:,0:3], cv.COLOR_BGR2GRAY)
    return cv.blur(img, (1,1))


def getWindowName(a):
    titles = set()
    def foo(hwnd,mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            titles.add(win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(foo, 0)
    lt = [t for t in titles if t]
    lt.sort()
    for t in lt:
        if(a in t and "Inspect" not in t):
            return t

def fftSnap(img):
    size = 10
    img = img
    (h, w) = img.shape
    (cX, cY) = (int(w/2.0), int(h/2.0))
    fft = np.fft.fft2(img)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean


def storePaperData(img, zs, fs):
    print(zs)
    print(fs)
    plt.scatter(zs, fs)
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.xlabel("z轴绝对位置")
    plt.ylabel("清晰度评价函数值")
    plt.show()
    try:
        os.mkdir("paperData")
    except Exception as e:
        print(e)
    try:
        os.mkdir("./paperData/img")
    except Exception as e:
        print(e)
    with open("./paperData/focusPlot.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(zs)
        writer.writerow(fs)
    for i in range(len(zs)):
        cv.imwrite("./paperData/img/" + str(zs[i])[:8].replace(".", "_") + ".tif", img[i])

def pygameLive():
    exitFlag = False
    pygame.init()
    display = pygame.display.set_mode((800, 800))
    imgSurf = pygame.surfarray.make_surface(np.zeros((700, 700), dtype=np.uint8))
    fps = 60
    liveClock = pygame.time.Clock()
    while not exitFlag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitFlag = True
        tmpImg = getLiveGrab()
        imgSurf = pygame.surfarray.make_surface(tmpImg)
        display.blit(imgSurf,(50, 50))
        pygame.display.update()
        liveClock.tick(fps)
    pygame.quit()

def setZ(z, core):
    core.set_position(z)
    while True:
        dis = abs(core.get_position() - z)
        if dis < 0.1:
            break

def testSetZ():
    while True:
        mes = int(input())
        setZ(mes, core)
        print(core.get_position())

def autoFocus():
    originZ = core.get_position()
    maxLap = 0
    maxZ = originZ
    delta = 200
    points = [
        [1024, 1024],
        [600, 600],
        [1400, 1400],
        [600, 1400],
        [1400, 600]
    ]
    quans = [
        1,
        0,
        0,
        0,
        0
    ]
    quickPanduan = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in [-0.5*x for x in range(0,50,1)]:
        setZ(originZ+i, core)
        time.sleep(0.1)
        snapImg = getSnap()
        blurCurrent = calculateValue(points, quans, snapImg, delta)
        # blurCurrent = calculateLaplacian(snapImg, [1024, 1024], 200)
        quickPanduan.append(blurCurrent)
        quickPanduan.pop(0)
        if not 0 in quickPanduan and max(quickPanduan) == quickPanduan[0]:
            break
        print(originZ + i, blurCurrent)
        if blurCurrent > maxLap:
            maxLap = blurCurrent
            maxZ = originZ+i
    setZ(originZ, core)
    quickPanduan = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in [-0.5*x for x in range(0,50,1)]:
        setZ(originZ-i, core)
        time.sleep(0.1)
        snapImg = getSnap()
        blurCurrent = calculateValue(points, quans, snapImg, delta)
        # blurCurrent = calculateLaplacian(snapImg, [1024, 1024], 200)
        quickPanduan.append(blurCurrent)
        quickPanduan.pop(0)
        if not 0 in quickPanduan and max(quickPanduan) == quickPanduan[0]:
            break
        print(originZ - i, blurCurrent)
        if blurCurrent > maxLap:
            maxLap = blurCurrent
            maxZ = originZ-i
    print("best",maxZ, maxLap)
    # setZ(maxZ, core)
    imgList = []
    zsList = []
    fsList = []
    setZ(maxZ - 10, core)
    zrange = 10
    steps = 100
    for z in [maxZ - zrange/2 + zrange/steps*x for x in range(steps)]:
        setZ(z, core)
        time.sleep(0.1)
        snapImg = getSnap()
        blurCurrent = calculateValue(points, quans, snapImg, delta)
        # blurCurrent = calculateLaplacian(snapImg, [1024, 1024], 200)
        imgList.append(snapImg)
        zsList.append(z)
        fsList.append(blurCurrent)
    storePaperData(imgList, zsList, fsList)
    for i in floatRange(core.get_position(), maxZ, 1):
        setZ(i, core)
        time.sleep(0.1)


def floatRange(start, end, step):
    re = []
    if start < end:
        while start + step < end:
            re.append(start + step)
            start = start + step
    else:
        while start - step > end:
            re.append(start - step)
            start = start - step
    return re

def getSnap():
    live = studio.live()
    live.snap(True)
    taggedImaged = core.get_tagged_image()
    pixels = np.reshape(
        taggedImaged.pix,
        newshape=[taggedImaged.tags['Height'], taggedImaged.tags['Width']]
        )
    return pixels

def calculateLaplacian(img, point, delta):
    x = point[0]
    y = point[1]
    # return cv.Laplacian(img[x-delta:x+delta, y-delta:y+delta], -1).var()
    return fftSnap(img[x-delta:x+delta, y-delta:y+delta])

def calculateValue(pointList, quanList, img, delta):
    re = 0
    for i in range(len(pointList)):
        tmp = calculateLaplacian(img, pointList[i], delta)
        re += tmp*quanList[i]
    return re

def autoFocusThisPy():
    while True:
        mes = input("请输入指令:")
        if mes == "1":
            autoFocus()
            # originZ = core.get_position()
            # maxLap = 0
            # maxZ = originZ
            # delta = 100
            # points = [
            #     [1024, 1024],
            #     [600, 600],
            #     [1400, 1400],
            #     [600, 1400],
            #     [1400, 600]
            # ]
            # quans = [
            #     1,
            #     0,
            #     0,
            #     0,
            #     0
            # ]
            # for i in [0.5*x for x in range(0,30,1)]:
            #     setZ(originZ+i, core)
            #     time.sleep(0.5)
            #     snapImg = getSnap()
            #     blurCurrent = calculateValue(points, quans, snapImg, delta)
            #     print(originZ + i, blurCurrent)
            #     if blurCurrent > maxLap:
            #         maxLap = blurCurrent
            #         maxZ = originZ+i
            # setZ(originZ, core)
            # for i in [0.5*x for x in range(0,30,1)]:
            #     setZ(originZ-i, core)
            #     time.sleep(0.5)
            #     snapImg = getSnap()
            #     blurCurrent = calculateValue(points, quans, snapImg, delta)
            #     print(originZ - i, blurCurrent)
            #     if blurCurrent > maxLap:
            #         maxLap = blurCurrent
            #         maxZ = originZ-i
            # print("best",maxZ, maxLap)
            # # setZ(maxZ, core)
            # for i in floatRange(core.get_position(), maxZ, 1):
            #     setZ(i, core)
            #     time.sleep(0.5)



def getLiveGrab():
    liveWindowName = getWindowName("Preview")
    handle = win32gui.FindWindow(0, liveWindowName)
    if handle != 0:
        currentRegion=win32gui.GetWindowRect(handle)
    img = grab_screen(currentRegion)
    return img

def sobelTest():
    img = getSnap()
    dstX = cv.Sobel(img, cv.CV_16U, 1, 0)
    dstY = cv.Sobel(img, cv.CV_16U, 0, 1)
    plt.subplot(1, 2, 1)
    plt.imshow(dstX, "gray")
    plt.subplot(1, 2, 2)
    plt.imshow(dstY, "gray")
    plt.show()
    

if __name__ == "__main__":
    bridge = Bridge()
    core = bridge.get_core()
    studio = bridge.get_studio()
    if __name__ == "__main__":
        autoFocusThisPy()