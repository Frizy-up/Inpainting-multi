import os
import shutil

# curWorkDir = os.getcwd()
# print curWorkDir
# os.chdir(curWorkDir)

# filePath = '/home/lab/Program-opt/MultiCamera/Dataset/Zip/test'

# filePath = '/home/lab/Program-opt/MultiCamera/Dataset/Zip/Left'
# resultFile = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/Left/'

# filePath = '/home/lab/Program-opt/MultiCamera/Dataset/Zip/Right'
# resultFile = '/home/lab/Program-opt/MultiCamera/DataSet-Mul/Right/'


def eachFile(filePath):

    pathDir = os.listdir(filePath)
    pathDir.sort()
    print pathDir

    for dir in pathDir:
        curDir = os.path.join(filePath,dir)
        print curDir

        if os.path.isfile(curDir):
            if os.path.splitext(curDir)[1] == ".png":
                print 'Process Files:', curDir
                textList = curDir.split('/')

                resultFileName = resultFile + textList[-2] + '-' + textList[-1]
                print resultFileName

                shutil.copy(curDir,resultFileName)


        else:
            eachFile(curDir)


eachFile(filePath)