from audioop import mul
import binascii
from ctypes.wintypes import tagRECT
from enum import auto
from glob import glob
import imp
import math
from pkgutil import extend_path
import threading
from socket import timeout
from subprocess import check_output
import time
from unittest import TextTestResult
# from pandas import ExcelWriter
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import N, messagebox
from tkinter.constants import BROWSE
from turtle import numinput
from cv2 import CAP_PROP_OPENNI_OUTPUT_MODE, adaptiveThreshold
from matplotlib.pyplot import step, text
from pycromanager import Bridge
from tkinter import ttk 
import json
import numpy as np
import cv2 as cv
import os
from PIL import ImageTk, Image
# from paddleocr import PaddleOCR
from zmq import curve_public
from autoFocus import autoFocus
from xyAdjust import multiPointMatch, uint16To8ThenOtsu

# ocr = PaddleOCR(lang="en",use_gpu=True)

def rotateImg(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(image, M, (nW, nH),borderValue=0)


def adjustImg(img):
    imgVis = img
    h, w = img.shape
    imgVis = imgVis * int(255/np.max(imgVis))
    # ret, imgVisTh = cv.threshold(imgVis, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # imgVisBlur = cv.GaussianBlur(imgVisTh, (15,15), 0)
    # edgeVis = cv.Canny(imgVisBlur, 10, 50)
    # lines = cv.HoughLinesP(edgeVis, 1, np.pi/180, 30, minLineLength=2000, maxLineGap=400)
    # rhos = []
    # for line in lines:
    #     x1 = line[0][0]
    #     y1 = line[0][1]
    #     x2 = line[0][2]
    #     y2 = line[0][3]
    #     rho = np.arctan((y2-y1)/(x2-x1))
    #     rhos.append(rho)
    # rhoMean = np.array(rhos).mean()
    # rotImg = rotateImg(imgVis, rhoMean)
    # ret = ocr.ocr(imgVis, cls=True)
    ret = 0 
    if len(ret) == 0:
        return 0, 0
    else:
        for i in ret:
            numInFlag = 0
            for j in range(10):
                if str(j+1) in i[1][0]:
                    numInFlag = 1
                    break
            if numInFlag:
                LTy = int(i[0][0][1])
                LTx = int(i[0][0][0])
                if LTy <= h/2:
                    return imgVis[LTy:LTy+1200,LTx-600:LTx+1000],[LTy,LTy+1200,LTx-600,LTx+1000]
                else:
                    return 0, 0
        if not numInFlag:
            return 0, 0
def str2hex(string):
    strBin = ("%s\r"%string).encode('utf-8')
    return bytes.fromhex(binascii.hexlify(strBin).decode('utf-8'))

class steper():
    def __init__(self) -> None:
        portList = list(serial.tools.list_ports.comports())
        if len(portList) < 1:
            messagebox.showerror(title="错误", message="无法连接电动载物台")
        else:
            self.name = portList[0].name
            self.bps = 19200
            self.timex = 0.01
        self.x = self.getX()
        self.y = self.getY()

    def send(self, mes):
        re = "E01"
        try:
            self.ser = serial.Serial(self.name, self.bps, timeout=self.timex)
            self.write(mes)
            re = self.read()
        except Exception as e:
            print(e)
        finally:
            try:
                self.ser.close()
            except Exception as e:
                print(e)
            return re

    def setZ(self,z):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_position(z)
            okFlag = 0
            for i in range(200):
                if abs(core.get_position() - z) < 0.5:
                    okFlag = 1
                    break
            if not okFlag:
                return "wrong"
            else:
                return "ok"
    
    def getZ(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            return core.get_position()
    
    def write(self, mes):
        self.ser.write(str2hex(mes))
    
    def read(self):
        for i in range(10):
            count = self.ser.inWaiting()
            if count != 0:
                break
            time.sleep(0.1)
        if count != 0:
            recv = self.ser.read(count)
            self.ser.flushInput()
            return recv.decode("utf-8")
        else:
            return -1

    def initX(self):
        re = self.send("position? x")
        if re != -1 and re != "E01":
            self.x = float(re.split(",")[1])
            return float(re.split(",")[1])
        else:
            return -1

    def initY(self):
        re = self.send("position? y")
        if re != -1 and re != "E01":
            self.y = float(re.split(",")[1])
            return float(re.split(",")[1])
        else:
            return -1


    def getX(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            return core.get_x_position()
        # re = self.send("position? x")
        # if re != -1 and re != "E01":
        #     before = self.x
        #     try:
        #         self.x = float(re.split(",")[1])
        #         return float(re.split(",")[1])
        #     except Exception as e:
        #         print(e)
        #         return before
        # else:
        #     return -1

    def getY(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            return core.get_y_position()
        # re = self.send("position? y")
        # if re != -1 and re != "E01":
        #     before = self.y
        #     try:
        #         self.y = float(re.split(",")[1])
        #         return float(re.split(",")[1])
        #     except Exception as e:
        #         print(e)
        #         return before
        # else:
        #     return -1

    def checkPos(self, zhou, v):
        for i in range(200):
            if zhou == "x":
                if self.getX() == v:
                    break
            else:
                if self.getY() == v:
                    break
            time.sleep(0.1)
        if zhou == "x":
            if self.getX() != v:
                return "wrong"
        else:
            if self.getY() != v:
                return "wrong"
        return "ok" 
                

    def xMoveAbsolute(self, x):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_xy_position("XYStage", x, self.getY())
            for i in range(100):
                if abs(self.getX() - x) < 0.5:
                    self.x = x
                    break
                time.sleep(0.1)
        # if x >= 0:
        #     dir = "p"
        # else:
        #     dir = "n"
        # if x > 10000 or x < -10000:
        #     return "wrong"
        # else:
        #     res = self.send("goposition x,o,a,%s,%f"%(dir, abs(x)))
        #     if res != -1 and res != "E01":
        #         if self.checkPos("x", x) == "ok":
        #             self.x = self.getX()
        #             return self.x
        #         else:
        #             self.x = self.getX()
        #             return "wrong"
        #     else:
        #         return "wrong"
    
    def yMoveAbsolute(self, y):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_xy_position("XYStage", self.getX(), y)
            for i in range(100):
                if abs(self.getY() - y) < 0.5:
                    self.y = y
                    break
                time.sleep(0.1)
        # if y >= 0:
        #     dir = "p"
        # else:
        #     dir = "n"
        # if y > 10000 or y < -10000:
        #     return "wrong"
        # else:
        #     res = self.send("goposition y,o,a,%s,%f"%(dir, abs(y)))
        #     if res != -1 and res != "E01":
        #         if self.checkPos("y", y) == "ok":
        #             self.y = self.getY()
        #             return self.y
        #         else:
        #             self.y = self.getY()
        #             return "wrong"
        #     else:
        #         return -1
    
    def xMoveRelative(self, x):
        target = self.getX() + x
        return self.xMoveAbsolute(target)
    
    def yMoveRelative(self, y):
        target = self.getY() + y
        return self.yMoveAbsolute(target)

class guiManager():
    def __init__(self) -> None:
        try:
            self.steper = steper()
        except Exception as e:
            print("未打开micromanager")
        self.data = dataManager()
        self.root = tk.Tk()
        self.root.geometry("1200x880+0+0")

        self.pointSelectFrame = tk.LabelFrame(
            master=self.root,
            text="Point",
            relief="groove",
            bd=5,
            height=880,
            width=360
        )       
        self.pointSelectFrame.grid(row=0, rowspan=6, column=0, columnspan=9,sticky="nw")

        self.posControlerFrame = tk.LabelFrame(
            master=self.root,
            text="Position control",
            relief="groove",
            bd=5,
            height=800,
            width=360
        )
        self.posControlerFrame.grid(row=7, rowspan=15, column=0, columnspan=9, sticky="nw")

        self.snapOptionsFrame = tk.LabelFrame(
            master=self.root,
            text="Camera",
            relief="groove",
            bd=5,
            height=40,
            width=880
        )
        self.snapOptionsFrame.grid(row=0,column=9,columnspan=30,sticky="nw")

        self.tifViewFrame = tk.LabelFrame(
            master=self.root,
            text="Image view",
            relief="groove",
            bd=5,
            height=840,
            width=880
        )
        self.tifViewFrame.grid(row=1, rowspan=21, column=9, columnspan=30,sticky="nw")
        
        self.subFrames= []

        self.initSnapOptionsFrame()
        self.initPointSelectFrame()
        try:
            self.initTifViewFrame()
        except Exception as e:
            print(e)
        self.initPosControlerFrame()
        self.root.bind('<Key>', self.rootKeyRes)
        self.time = time.time()
    
    def rootKeyRes(self, key):
        now = time.time()
        slowStep = float(self.slowMoveStepEntry.get())
        fastStep = float(self.fastMoveStepEntry.get())
        if self.manualMoveMode.get() == 1:
            keyChar = key.char
            if keyChar >= "a":
                if now - self.time > slowStep/5*0.5:
                    self.time = now
                    if keyChar == "w":
                        self.steper.xMoveRelative(slowStep)
                    elif keyChar == "s":
                        self.steper.xMoveRelative(-slowStep)
                    elif keyChar == "a":
                        self.steper.yMoveRelative(slowStep)
                    elif keyChar == "d":
                        self.steper.yMoveRelative(-slowStep)
            if keyChar >= "A" and keyChar < 'a':
                if now - self.time > fastStep/100*8:
                    if keyChar == "W":
                        self.steper.xMoveRelative(fastStep)
                    elif keyChar == "S":
                        self.steper.xMoveRelative(-fastStep)
                    elif keyChar == "A":
                        self.steper.yMoveRelative(fastStep)
                    elif keyChar == "D":
                        self.steper.yMoveRelative(-fastStep)


    def createSubframe(self,parFrame, row, column, rowspan, columnspan, h, w):
        self.subFrames.append(tk.Frame(master=parFrame, height=h, width=w))
        self.subFrames[-1].grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan)
        self.subFrames[-1].pack_propagate(0)

    def setBF(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_config("0 Channels", "BF-1")
    
    def setTRITC(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_config("0 Channels", "TRITC")

    def setFITC(self):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_config("0 Channels", "FITC")
    
    def recordPosition(self):
        currentSet = self.data.config["currentPointSet"]
        currentPoint = self.data.config["currentPoint"]
        # 第一次拍照记录，其后不记录
        # try:
        #     test = self.data.config["positionSets"][currentSet][currentPoint]
        # except KeyError as e:
        #     self.data.addPosition(currentSet,
        #                           currentPoint,
        #                           self.steper.getX(),
        #                           self.steper.getY(),
        #                           self.steper.getZ()
        #                           )

        # 每次拍照位置更新
        self.data.addPosition(currentSet,
                              currentPoint,
                              self.steper.getX(),
                              self.steper.getY(),
                              self.steper.getZ()
                              )
 
 
    def snapAndRecord(self):
        self.snap()
        # self.recordPosition()
    
    def startLoopThread(self):
        loopTh = threading.Thread(target=self.startLoop)
        loopTh.start()
    
    def startLoop(self):
        gap = float(self.loopGapEntry.get())
        times = int(self.loopTimesEntry.get())
        for i in range(times):
            # n = 0
            # while True:
            #     n += 1
            #     currentPointSet = self.data.config["currentPointSet"]
            #     currentPoint = self.data.config["currentPoint"]
            #     x,y,z = self.data.config["positionSets"][currentPointSet][currentPoint]
            #     print("target:%f,%f,%f"%(x,y,z))
            #     if n > 1:
            #         if self.lastPoint == currentPoint:
            #             break
            #         else:
            #             if currentPoint == "A1":
            #                 break
            #             else:
            #                 self.lastPoint = currentPoint
            #     self.steper.setZ(z)
            #     self.steper.xMoveAbsolute(x)
            #     self.steper.yMoveAbsolute(y)
            #     self.snap()
            currentPointSet = self.data.config["currentPointSet"]
            currentPoint = self.data.config["currentPoint"]
            currentPointList = self.data.config["pointSets"][currentPointSet]
            currentIndex = currentPointList.index(currentPoint)
            for p in currentPointList[currentIndex:]:
                x,y,z = self.data.config["positionSets"][currentPointSet][p]
                print("target:%f,%f,%f"%(x,y,z))
                self.steper.setZ(z)
                self.steper.xMoveAbsolute(x)
                self.steper.yMoveAbsolute(y)
                self.snap()
            if self.autoMoveMode.get() == 0:
                break
            time.sleep(gap*60)

    def snap(self):
        global DIR
        with Bridge() as bridge:
            core = bridge.get_core()
            studio = bridge.get_studio()
            live = studio.live()
            if self.autoMoveMode.get() == 1:
                autoFocus()
                # if self.data.config["currentTime"] >= 0:
                #     path = DIR + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
                #     for i in os.listdir(path):
                #         if int(i.split("-")[0]) == self.data.config["currentTime"]:
                #             nameBF = i
                #             break
                #     beforeTif = cv.imread(path + nameBF, -1)
                #     beforeTif = uint16To8ThenOtsu(beforeTif)
                #     live.snap(True)
                #     img = core.get_tagged_image()
                #     currentTif = np.reshape(img.pix, newshape=[img.tags["Height"], img.tags["Width"]])
                #     currentTif = uint16To8ThenOtsu(currentTif)
                #     dx, dy = multiPointMatch(beforeTif, [[1000,1000]], 300, currentTif)
                #     if dx < 2048/4:
                #         self.steper.yMoveRelative(-dx * 0.065)
                #     if dy < 2048/4:
                #         self.steper.xMoveRelative(dy * 0.065)
        self.recordPosition()

        time.sleep(0.5)
        with Bridge() as bridge:
            core = bridge.get_core()
            studio = bridge.get_studio()
            live = studio.live()
            if self.pointSetSelectCombobox.get() == "default":
                messagebox.showerror(title="错误", message="不能选择default点集")
                return 0
            if live.is_live_mode_on():
                if self.BF.get():
                    core.set_config("0 Channels", "BF-1")
                    time.sleep(1.5)
                    live.snap(True)
                    img = core.get_tagged_image()
                    img = np.reshape(img.pix, newshape=[img.tags["Height"], img.tags["Width"]])
                    path = DIR + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
                    tifNames = os.listdir(path)
                    name = str(len(tifNames)) + "-" + time.asctime().replace(" ", "-").replace(":", "-") + ".tif"
                    cv.imwrite(path+name, img)
                if self.FITC.get():
                    core.set_config("0 Channels", "FITC")
                    time.sleep(1.5)
                    live.snap(True)
                    img = core.get_tagged_image()
                    img = np.reshape(img.pix, newshape=[img.tags["Height"], img.tags["Width"]])
                    path = DIR + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\FITC\\"
                    tifNames = os.listdir(path)
                    name = str(len(tifNames)) + "-" + time.asctime().replace(" ", "-").replace(":", "-") + ".tif"
                    cv.imwrite(path+name, img)
                if self.TRITC.get():
                    core.set_config("0 Channels", "TRITC")
                    time.sleep(1.5)
                    live.snap(True)
                    img = core.get_tagged_image()
                    img = np.reshape(img.pix, newshape=[img.tags["Height"], img.tags["Width"]])
                    path = DIR + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\TRITC\\"
                    tifNames = os.listdir(path)
                    name = str(len(tifNames)) + "-" + time.asctime().replace(" ", "-").replace(":", "-") + ".tif"
                    cv.imwrite(path+name, img)
                core.set_config("0 Channels", "BF-1")
                pointList = self.data.config["pointSets"][self.data.config["currentPointSet"]]
                currentIndex = pointList.index(self.data.config["currentPoint"])
                if currentIndex + 1 < len(pointList):
                    nextPoint = pointList[currentIndex+1]
                    self.pointSelectCombobox.set(nextPoint)
                    self.data.selectPoint(nextPoint)
                    self.goToCurrentPosition()
                    self.showLastTif()
                else:
                    if self.autoMoveMode.get() == 1:
                        nextPoint = pointList[0]
                        self.pointSelectCombobox.set(nextPoint)
                        self.data.selectPoint(nextPoint)
                        self.goToCurrentPosition()
                        self.showLastTif()
            else:
                messagebox.showerror(title="错误",message="Live未打开")

    def setTimeZero(self):
        self.data.setTimeZero()
        self.currentTimeLabel.configure(text="0")

    def initTifViewFrame(self):

        if self.data.config["currentPointSet"] != "default":
            pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
            pathFITC = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\FITC\\"
            nameBF = ""
            if len(os.listdir(pathBF)) > 0:
                for i in os.listdir(pathBF):
                    if int(i.split("-")[0]) == self.data.config["currentTime"]:
                        nameBF = i
                        break
                nameFITC = ""
                for i in os.listdir(pathFITC):
                    if int(i.split("-")[0]) == self.data.config["currentTime"]:
                        nameFITC = i
                        break
                img = cv.imread(pathBF + nameBF, 0)
                try:
                    img = cv.resize(img, (800,800))
                    self.currentTif = ImageTk.PhotoImage(image=Image.fromarray(img))
                except Exception as e:
                    print(e)
                try:
                    self.tifLabel = tk.Label(self.tifViewFrame, image=self.currentTif)
                except Exception as e:
                    print(e)
                    self.tifLabel = tk.Label(self.tifViewFrame, image=ImageTk.PhotoImage(image=Image.fromarray(np.zeros((500,500)))))
                self.tifLabel.pack()
    
    def showCurrentTif(self):
        if self.data.config["currentPointSet"] != "default":
            pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
            pathFITC = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\FITC\\"
            pathTRITC = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\TRITC\\"
            nameBF = ""
            if len(os.listdir(pathBF)) > 0:
                for i in os.listdir(pathBF):
                    if int(i.split("-")[0]) == self.data.config["currentTime"]:
                        nameBF = i
                        break
                nameFITC = ""
                for i in os.listdir(pathFITC):
                    if int(i.split("-")[0]) == self.data.config["currentTime"]:
                        nameFITC = i
                        break
                nameTRITC = ""
                for i in os.listdir(pathTRITC):
                    if int(i.split("-")[0]) == self.data.config["currentTime"]:
                        nameTRITC = i
                        break
                img = cv.imread(pathBF + nameBF, 0)
                if self.isAdjust.get():
                    adImg, td = adjustImg(img)
                if self.isShowBF.get():
                    if self.isAdjust.get():
                        if type(adImg) == np.ndarray:
                            img = adImg * 2
                    else:
                        img = img * 2
                elif self.isShowFITC.get():
                    if self.isAdjust.get():
                        if type(adImg) == np.ndarray:
                            img = cv.imread(pathFITC + nameFITC)
                            if np.max(img) != 0:
                                factor = int(255/np.max(img))
                            else:
                                factor = 1
                            img = factor * img[td[0]:td[1],td[2]:td[3]]
                    else:
                        img = cv.imread(pathFITC + nameFITC, 0)
                        if np.max(img) != 0:
                            factor = int(255/np.max(img))
                        else:
                            factor = 1
                        img = factor * img
                elif self.isShowTRITC.get():
                    if self.isAdjust.get():
                        if type(adImg) == np.ndarray:
                            img = cv.imread(pathTRITC + nameTRITC)
                            if np.max(img) != 0:
                                factor = int(255/np.max(img))
                            else:
                                factor = 1
                            img = factor * img[td[0]:td[1],td[2]:td[3]]
                    else:
                        img = cv.imread(pathTRITC + nameTRITC, 0)
                        if np.max(img) != 0:
                            factor = int(255/np.max(img))
                        else:
                            factor = 1
                        img = factor * img
                else:
                    if self.isAdjust.get():
                        img = adImg
                    else:
                        img = img
                try:
                    img = cv.resize(img, (800,800))
                except Exception as e:
                    print(e)
                print(img)
                self.currentTif = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.tifLabel.configure(image=self.currentTif)

    def initSnapOptionsFrame(self):
        self.BF = tk.IntVar()
        self.BFcheck = tk.Checkbutton(self.snapOptionsFrame, text="BF-1", variable=self.BF)
        self.BFcheck.pack(side="left")
        self.FITC = tk.IntVar()
        self.FITCcheck = tk.Checkbutton(self.snapOptionsFrame, text="FITC", variable=self.FITC)
        self.FITCcheck.pack(side="left")
        self.TRITC = tk.IntVar()
        self.TRITCcheck = tk.Checkbutton(self.snapOptionsFrame, text="TRITC", variable=self.TRITC)
        self.TRITCcheck.pack(side="left")
        self.snapButton = tk.Button(self.snapOptionsFrame, text="Album", command=self.snapAndRecord)
        self.snapButton.pack(side="left")
        self.liveBfButton = tk.Button(self.snapOptionsFrame, text="liveBF-1", command=self.setBF)
        self.liveBfButton.pack(side="left")
        self.liveFitcButton = tk.Button(self.snapOptionsFrame, text="liveFITC", command=self.setFITC)
        self.liveFitcButton.pack(side="left")
        self.liveTritcButton = tk.Button(self.snapOptionsFrame, text="liveTRITC", command=self.setTRITC)
        self.liveTritcButton.pack(side="left")

    def initPosControlerFrame(self):
        self.createSubframe(self.posControlerFrame, 0,0,1,9,35,315)
        self.autoMoveMode = tk.IntVar()
        self.autoMoveModeCheck = tk.Checkbutton(self.subFrames[-1], text="Automatic mode", variable=self.autoMoveMode, command=self.checkAuto)
        self.autoMoveModeCheck.pack(side="left")

        self.createSubframe(self.posControlerFrame, 1,0,1,9,35,135)
        self.autoMoveStartButton = tk.Button(self.subFrames[-1], text="Start traversal", command=self.startLoopThread)
        self.autoMoveStartButton.pack(side="right")


        self.createSubframe(self.posControlerFrame, 2,0,1,9,35,135)
        self.loopGapLabel = tk.Label(self.subFrames[-1], text="Interval")
        self.loopGapLabel.pack(side="left")
        self.loopGapEntry = tk.Entry(self.subFrames[-1], width=3, bd=2)
        self.loopGapEntry.pack(side="left")
        self.loopTimesLabel = tk.Label(self.subFrames[-1], text="Num")
        self.loopTimesLabel.pack(side="left")
        self.loopTimesEntry = tk.Entry(self.subFrames[-1], width=3, bd=2)
        self.loopTimesEntry.pack(side="left")
        # self.autoMoveStopButton = tk.Button(self.subFrames[-1], text="停止遍历")
        # self.autoMoveStopButton.pack(side="left")

        self.createSubframe(self.posControlerFrame, 3,0,1,9,35,315)
        self.manualMoveMode = tk.IntVar()
        self.manualMoveModeCheck = tk.Checkbutton(self.subFrames[-1], text="Manual mode", variable=self.manualMoveMode, command=self.checkManual)
        self.manualMoveModeCheck.pack(side="left")

        self.createSubframe(self.posControlerFrame, 4,0,1,9,35,315)
        self.fastMoveStepLabel = tk.Label(self.subFrames[-1], text="Fast step size:")
        self.fastMoveStepLabel.pack(side="left")
        self.fastMoveStepEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.fastMoveStepEntry.insert(0, "100")
        self.fastMoveStepEntry.pack(side="left")
        self.fastMoveStepEntry.bind("<Return>",lambda:1)

        self.createSubframe(self.posControlerFrame, 5,0,1,9,35,315)
        self.slowMoveStepLabel = tk.Label(self.subFrames[-1], text="Slow step size:")
        self.slowMoveStepLabel.pack(side="left")
        self.slowMoveStepEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.slowMoveStepEntry.insert(0, "5")
        self.slowMoveStepEntry.pack(side="left")
        self.slowMoveStepEntry.bind("<Return>",lambda:1)

        self.createSubframe(self.posControlerFrame, 6,0,1,9,35,135)
        self.absoluteMode = tk.IntVar()
        self.absoluteModeCheck = tk.Checkbutton(self.subFrames[-1], text="Absolute pos. mode", variable=self.absoluteMode, command=self.checkAbsolute)
        self.absoluteModeCheck.pack(side="left")

        self.createSubframe(self.posControlerFrame, 7,0,1,9,35,135)
        self.xAbsoluteLabel = tk.Label(self.subFrames[-1], text="x:")
        self.xAbsoluteLabel.pack(side="left")
        self.xAbsoluteEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.xAbsoluteEntry.pack(side="left")

        self.createSubframe(self.posControlerFrame, 8,0,1,9,35,135)
        self.yAbsoluteLabel = tk.Label(self.subFrames[-1], text="y:")
        self.yAbsoluteLabel.pack(side="left")
        self.yAbsoluteEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.yAbsoluteEntry.pack(side="left")

        self.createSubframe(self.posControlerFrame, 9,0,1,9,35,135)
        self.absoluteModeStartButton = tk.Button(self.subFrames[-1], text="Go", command=self.absoluteGo)
        self.absoluteModeStartButton.pack(side="left")

        self.createSubframe(self.posControlerFrame, 10,0,1,9,35,135)
        self.relativeMode = tk.IntVar()
        self.relativeModeCheck = tk.Checkbutton(self.subFrames[-1], text="Relative pos. mode", variable=self.relativeMode, command=self.checkRelative)
        self.relativeModeCheck.pack(side="left")

        self.createSubframe(self.posControlerFrame, 11,0,1,9,35,135)
        self.xRelativeLabel = tk.Label(self.subFrames[-1], text="x:")
        self.xRelativeLabel.pack(side="left")
        self.xRelativeEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.xRelativeEntry.pack(side="left")

        self.createSubframe(self.posControlerFrame, 12,0,1,9,35,135)
        self.yRelativeLabel = tk.Label(self.subFrames[-1], text="y:")
        self.yRelativeLabel.pack(side="left")
        self.yRelativeEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.yRelativeEntry.pack(side="left")

        self.createSubframe(self.posControlerFrame, 13,0,1,9,35,135)
        self.relativeModeStartButton = tk.Button(self.subFrames[-1], text="Go", command=self.relativeGo)
        self.relativeModeStartButton.pack(side="left")


    def initPointSelectFrame(self):
        self.createSubframe(self.pointSelectFrame,0,0,1,9,35,315)
        self.newPointSetLabel = tk.Label(self.subFrames[-1], text="New point set")
        self.newPointSetLabel.pack(side="left")
        self.newPointSetEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.newPointSetEntry.pack(side="left")
        self.newPointSetEntry.bind("<Return>", self.newPointSet)

        self.createSubframe(self.pointSelectFrame,1,0,1,2,35,75)
        self.pointSetSelectLabel = tk.Label(self.subFrames[-1],text="point set", relief="groove")
        self.pointSetSelectLabel.pack(side="left")

        self.createSubframe(self.pointSelectFrame,1,2,1,6,35,210)
        self.pointSetSelectCombobox = ttk.Combobox(self.subFrames[-1])
        self.pointSetSelectCombobox["values"] = list(self.data.config["pointSets"].keys())
        self.pointSetSelectCombobox.bind("<<ComboboxSelected>>", self.pointSetSelectEvent)
        self.pointSetSelectCombobox.current(list(self.data.config["pointSets"].keys()).index(self.data.config["currentPointSet"]))
        self.pointSetSelectCombobox.pack(fill="both")

        self.createSubframe(self.pointSelectFrame,2,0,1,9,35,315)
        self.newPointLabel = tk.Label(self.subFrames[-1], text="New point")
        self.newPointLabel.pack(side="left")
        self.newPointEntry = tk.Entry(self.subFrames[-1], bd=2)
        self.newPointEntry.pack(side="left")
        self.newPointEntry.bind("<Return>", self.newPoint)

        self.createSubframe(self.pointSelectFrame,3,0,1,2,35,75)
        self.pointSelectLabel = tk.Label(self.subFrames[-1],text="point", relief="groove")
        self.pointSelectLabel.pack(side="left")

        self.createSubframe(self.pointSelectFrame,3,2,1,6,35,210)
        self.pointSelectCombobox = ttk.Combobox(self.subFrames[-1])
        self.pointSelectCombobox["values"] = self.data.config["pointSets"][self.data.config["currentPointSet"]]
        self.pointSelectCombobox.bind("<<ComboboxSelected>>", self.pointSelectEvent)
        if self.data.config["currentPointSet"] !="default":
            self.pointSelectCombobox.current(self.data.config["pointSets"][self.data.config["currentPointSet"]].index(self.data.config["currentPoint"]))
            self.lastPoint = self.pointSelectCombobox.get()
        self.pointSelectCombobox.pack(fill="both")

        self.createSubframe(self.pointSelectFrame,4,0,1,9,35,210)
        self.leftButton = tk.Button(self.subFrames[-1], text="<<", command=self.showPreviousTif)
        self.leftButton.pack(side="left")
        self.currentTimeLabel = tk.Label(self.subFrames[-1], text=str(self.data.config["currentTime"]))
        self.currentTimeLabel.pack(side="left")
        self.rightButton = tk.Button(self.subFrames[-1], text=">>", command=self.showNextTif)
        self.rightButton.pack(side="left")
        self.isAdjust = tk.IntVar()
        self.adjustCheck = tk.Checkbutton(self.subFrames[-1], text="XY Adj.", variable=self.isAdjust, command=self.showCurrentTif)
        self.adjustCheck.pack(side="left")
        self.isShowBF = tk.IntVar()
        self.showBFCheck = tk.Checkbutton(self.subFrames[-1], text="BF", variable=self.isShowBF, command=self.showCurrentTif)
        self.showBFCheck.pack(side="left")
        self.isShowFITC = tk.IntVar()
        self.showFITCCheck = tk.Checkbutton(self.subFrames[-1], text="FITC", variable=self.isShowFITC, command=self.showCurrentTif)
        self.showFITCCheck.pack(side="left")

        self.createSubframe(self.pointSelectFrame,5,0,1,9,35,210)
        self.viewInImageJButton = tk.Button(self.subFrames[-1], text="Open with ImageJ", command=self.viewInImageJ)
        self.viewInImageJButton.pack(side="left")

        self.isShowTRITC = tk.IntVar()
        self.showTRITCCheck = tk.Checkbutton(self.subFrames[-1], text="TRITC", variable=self.isShowTRITC, command=self.showCurrentTif)
        self.showTRITCCheck.pack(side="left")

        self.goToButton = tk.Button(self.subFrames[-1], text="Go", command=self.goToCurrentPosition)
        self.goToButton.pack(side="left")

    def goToCurrentPosition(self):
        currentSet = self.data.config["currentPointSet"]
        currentPoint = self.data.config["currentPoint"]
        x, y, z = self.data.config["positionSets"][currentSet][currentPoint]
        if x != -1 and y != -1:
            self.goToPosition(x, y, z)
        for i in range(20):
            if (
                abs(x - self.steper.getX()) < 1 and
                abs(y - self.steper.getY()) < 1 and
                abs(z - self.steper.getZ()) < 1
                ):
                break
            time.sleep(0.5)

    
    def goToPosition(self, x, y, z):
        with Bridge() as bridge:
            core = bridge.get_core()
            core.set_xy_position("XYStage", x, y)
            core.set_position(z)

    def absoluteGo(self):
        if self.absoluteMode.get() == 1:
            x = float(self.xAbsoluteEntry.get())
            y = float(self.yAbsoluteEntry.get())
            if x != self.steper.x:
                self.steper.xMoveAbsolute(x)
            if y != self.steper.y:
                self.steper.yMoveAbsolute(y)
    
    def relativeGo(self):
        if self.relativeMode.get() == 1:
            x = float(self.xRelativeEntry.get())
            y = float(self.yRelativeEntry.get())
            self.steper.xMoveRelative(x)
            self.steper.yMoveRelative(y)

    def checkAuto(self):
        if self.autoMoveMode.get()==1:
            self.manualMoveMode.set(0)
            self.absoluteMode.set(0)
            self.relativeMode.set(0)

    def checkManual(self):
        if self.manualMoveMode.get()==1:
            self.root.focus_set()
            self.autoMoveMode.set(0)
            self.absoluteMode.set(0)
            self.relativeMode.set(0)

    def checkAbsolute(self):
        if self.absoluteMode.get()==1:
            self.autoMoveMode.set(0)
            self.manualMoveMode.set(0)
            self.relativeMode.set(0)

    def checkRelative(self):
        if self.relativeMode.get()==1:
            self.autoMoveMode.set(0)
            self.manualMoveMode.set(0)
            self.absoluteMode.set(0)

    def viewInImageJ(self):
        pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
        pathFITC = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\FITC\\"
        pathTRITC = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\TRITC\\"
        nameBF = ""
        if len(os.listdir(pathBF)) > 0:
            for i in os.listdir(pathBF):
                if int(i.split("-")[0]) == self.data.config["currentTime"]:
                    nameBF = i
                    break
            nameFITC = ""
            for i in os.listdir(pathFITC):
                if int(i.split("-")[0]) == self.data.config["currentTime"]:
                    nameFITC = i
                    break
            nameTRITC = ""
            for i in os.listdir(pathTRITC):
                if int(i.split("-")[0]) == self.data.config["currentTime"]:
                    nameTRITC = i
                    break
 
        if nameBF != "":
            if self.isShowFITC.get():
                path = pathFITC + nameFITC
            elif self.isShowTRITC.get():
                path = pathTRITC + nameTRITC
            else:
                path = pathBF + nameBF
            os.system("start imageJ "+ "\"" + path + "\"")
    
    def showNextTif(self):
        pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
        le = len(os.listdir(pathBF))
        if self.data.config["currentTime"] + 1 < le:
            self.data.selectNextTime()
            self.currentTimeLabel.configure(text=str(self.data.config["currentTime"]))
            self.showCurrentTif()

    def showPreviousTif(self):
        pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
        le = len(os.listdir(pathBF))
        if self.data.config["currentTime"] - 1 >= 0:
            self.data.selectPreviousTime()
            self.currentTimeLabel.configure(text=str(self.data.config["currentTime"]))
            self.showCurrentTif()
    
    def showLastTif(self):
        pathBF = self.data.dir + self.data.config["currentPointSet"] + "\\" + self.data.config["currentPoint"] + "\\BF-1\\"
        le = len(os.listdir(pathBF))
        if le >= 0:
            self.data.selectTime(le-1)
            self.currentTimeLabel.configure(text=str(self.data.config["currentTime"]))
            # self.showCurrentTif()

    def pointSetSelectEvent(self,event):
        pointSet = self.pointSetSelectCombobox.get()
        self.data.selectPointSet(pointSet)
        self.pointSelectCombobox["values"] = self.data.config["pointSets"][self.data.config["currentPointSet"]]
        for i in self.pointSelectCombobox.get():
            self.pointSelectCombobox.delete(0)

    def pointSelectEvent(self,event):
        point = self.pointSelectCombobox.get()
        self.lastPoint = point
        self.data.selectPoint(point)
        self.showLastTif()

    def newPoint(self,event):
        name = self.newPointEntry.get()
        for i in range(len(name)):
            self.newPointEntry.delete(0)
        self.data.addPoint(self.data.config["currentPointSet"],name)
        self.pointSelectCombobox["values"] = self.data.config["pointSets"][self.data.config["currentPointSet"]]
        self.pointSelectCombobox.current()
        self.pointSelectEvent(None)

    def newPointSet(self,event):
        name = self.newPointSetEntry.get()
        for i in range(len(name)):
            self.newPointSetEntry.delete(0)
        self.data.addPointSets(name)
        self.data.addPositionSets(name)
        self.pointSetSelectCombobox["values"] = list(self.data.config["pointSets"].keys())

    def run(self):
        self.root.mainloop()


class dataManager():
    def __init__(self) -> None:
        self.dir = DIR
        self.initConfig()

    def initConfig(self):
        if not os.path.exists(self.dir+"config.json"):
            with open(self.dir + "config.json", "w+") as f:
                self.config = {
                    "pointSets":{"default":[]},
                    "positionSets":{"default":{}},
                    "currentPointSet":"default",
                    "currentPoint":"default",
                    "currentTime":0,
                    "currentViewChannel":"BF-1",
                    "currentSnapChannel":[]
                    }
                f.write(json.dumps(self.config, indent=4))
        else:
            with open(self.dir + "config.json", "r") as f:
                self.config = json.load(f)
    
    def saveConfig(self):
        with open(self.dir + "config.json", "w+") as f:
            f.write(json.dumps(self.config, indent=4))
    
    def addPointSets(self, setName):
        if not os.path.exists(self.dir + setName):
            os.mkdir(self.dir+setName)
            self.config["pointSets"][setName]=[]
            self.saveConfig()

    def addPositionSets(self, setName):
        self.config["positionSets"][setName]={}
        self.saveConfig()

    def selectNextTime(self):
        self.config["currentTime"] += 1
        self.saveConfig()
    
    def selectTime(self, t):
        self.config["currentTime"] = t
        self.saveConfig()
    
    def setTimeZero(self):
        self.config["currentTime"] = 0
        self.saveConfig()
    
    def selectPreviousTime(self):
        self.config["currentTime"] -= 1
        self.saveConfig()

    def selectPointSet(self, setName):
        self.config["currentPointSet"] = setName
        self.saveConfig()

    def addPoint(self, setName, pointName):
        if not os.path.exists(self.dir + setName + "\\" + pointName):
            os.mkdir(self.dir + setName + "\\" + pointName)
            os.mkdir(self.dir + setName + "\\" + pointName + "\\" + "BF-1")
            os.mkdir(self.dir + setName + "\\" + pointName + "\\" + "FITC")
            os.mkdir(self.dir + setName + "\\" + pointName + "\\" + "TRITC")
            self.config["pointSets"][setName].append(pointName)
            self.saveConfig()
    
    def addPosition(self, setName, positionName, x, y, z):
        self.config["positionSets"][setName][positionName] = [x, y, z]
        self.saveConfig()
    
    def selectPoint(self, pointName):
        self.config["currentPoint"] = pointName
        self.saveConfig()

DIR = os.path.dirname(__file__) + "\\"
if __name__ == "__main__":
    gui = guiManager()
    gui.run()
# with Bridge() as bridge:
#     bridge = Bridge()
#     core = bridge.get_core()
#     studio = bridge.get_studio()
#     live = studio.live()