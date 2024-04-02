import os
import trace
import traceback

import matplotlib.pyplot as plt
import pandas as pd


def getHgtCellId(colorPath):
    color = pd.read_csv(colorPath)
    re_ = {}
    for row in color.itertuples():
        mark = {}
        firstFlag = False
        colCount = len(row)
        for i in range(2, colCount-1):
            if not firstFlag:
                if row[i] == 1:
                    firstFlag = True
                    break
                if row[i] == 2:
                    firstFlag = True
                    mark["first"] = i-2
            if row[i] == 2 and row[i+1] == 1:
                if i < colCount-3:
                    if row[i+2] == 1 and row[i+3] == 1:
                        mark["turn"] = i+1-2
                        re_[row.id] = mark
                        break

    return re_

def readContactInfo(frameDir):
    dirs = os.listdir(frameDir)
    re_ = []
    for i in range(len(dirs)):
        filePath = frameDir + os.sep + "frame%d.csv"%i
        with open(filePath, "r") as f:
            lines = f.readlines()[1:]
        re_.append([list(map(int, l.replace("\n", "").split(","))) for l in lines if "," in l])
    return re_


def calculateNewHgt(colorPath, frameDir):
    hgtInfo = getHgtCellId(colorPath)
    contactInfo = readContactInfo(frameDir)
    frameCount = len(contactInfo)
    re_ = {}
    for k in hgtInfo.keys():
        tmp = hgtInfo[k]
        inf = tmp["first"]
        sup = tmp["turn"]
        if inf >= frameCount or sup >= frameCount:
            continue
        foundFlag = False
        for f in range(inf, sup):
            for g in contactInfo[f]:
                if k in g:
                    foundFlag = True
                    break
            if foundFlag:
                tmp["startContact"] = f
                break
        if foundFlag:
            re_[k] = tmp
    noHgtContactCells = []
    hgtCells = list(hgtInfo.keys())
    for f in contactInfo:
        for g in f:
            if len(g) > 1:
                for r in g[1:]:
                    if r not in hgtCells:
                        noHgtContactCells.append(r)
    noHgtContactCount = len(list(set(noHgtContactCells)))
    contactCount = noHgtContactCount + len(re_)
    valid = 0
    rate = 0
    for k in re_.keys():
        tmp = re_[k]
        bottom = tmp["turn"] - tmp["startContact"]
        if  bottom > 0:
            valid += 1/(0.5*bottom)
    #0.5 is the time interval (hour)
    rate = valid/contactCount

    # print the results of valid, contactCount, and rate
    print("Valid:", valid)
    print("contactCount:", contactCount)
    print("rate:", rate)


    return rate, re_

if __name__ == "__main__":

    colorPath = 'D:/li/20240218/1/color.csv'
    frameDir = 'D:/li/20240218/1/contact'


    try:
        hgt_result,re_data = calculateNewHgt(colorPath, frameDir)

        re_df = pd.DataFrame.from_dict(re_data, orient='index')
        re_df.to_csv('D:/li/20240218/1/re_conj.csv', index_label='id')


        with open('D:/li/20240218/1/HGTresults.csv', 'w') as file:
            file.write('HGT_Rate\n')
            file.write(str(hgt_result) + '\n')

    except Exception as e:
        print("An error occurred:", e)


