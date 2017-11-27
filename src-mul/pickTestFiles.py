import os
import shutil
import random

# leftFilePath = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/trainData/Left'
# resultLeftFilePath = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/testData/Left/'
#
#
# rightFilePath = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/trainData/Right'
# resultRightFilePath = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/testData/Right/'

def eachFile(leftFilePath, rightFilePath, resultLeftFilePath, resultRightFilePath):

    leftFiles = os.listdir(leftFilePath)
    rightFiles = os.listdir(rightFilePath)
    leftFiles.sort()
    rightFiles.sort()

    len_left = len(leftFiles)
    len_right = len(rightFiles)

    if len_left != len_right:
        raise AssertionError

    print len_left

    i = 0
    while i < len_left and i < len_right:

        rand = random.randint(0, 53)

        curLeftFilePath = os.path.join(leftFilePath, leftFiles[i+rand])
        curRightFilePath = os.path.join(rightFilePath,rightFiles[i+rand])
        print curLeftFilePath,curRightFilePath

        resultLeftFileName  = resultLeftFilePath  + leftFiles[i + rand]
        resultRightFileName = resultRightFilePath + rightFiles[i + rand]

        shutil.copy(curLeftFilePath,resultLeftFileName)
        os.remove(curLeftFilePath)

        shutil.copy(curRightFilePath, resultRightFileName)
        os.remove(curRightFilePath)

        i = i + 54



eachFile(leftFilePath, rightFilePath, resultLeftFilePath, resultRightFilePath)