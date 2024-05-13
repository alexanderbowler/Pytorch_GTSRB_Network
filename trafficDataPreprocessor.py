import os
from PIL import Image
import pandas as pd

#preprocessing the Train dataset
dataPath = "./curTrain/"
dimSize = 45
for label, dir in enumerate(os.listdir(dataPath)):
    for imgName in os.listdir(dataPath+dir):
        img = Image.open(dataPath+dir+"/"+imgName)
        imgWidth, imgHeight = img.size
        if(imgWidth >= dimSize
            and imgHeight >= dimSize and imgHeight/imgWidth < 1.1 and imgWidth/imgHeight < 1.1):
            img = img.resize((dimSize,dimSize))
            img.save(dataPath+dir+"/"+imgName)
        else:
            os.remove(dataPath+dir+"/"+imgName)


#preprocessing Test dataset
dataPath = "./curTest/"
for imgName in os.listdir(dataPath):
    img = Image.open(dataPath+imgName)
    imgWidth, imgHeight = img.size
    if(imgWidth >= dimSize and imgHeight >= dimSize and imgHeight/imgWidth < 1.1 and imgWidth/imgHeight < 1.1):
        img = img.resize((dimSize,dimSize))
        img.save(dataPath+imgName)
    else:
        os.remove(dataPath+imgName)


trainDf = pd.read_csv("Test.csv")
trainDf = trainDf[trainDf["Height"] >= dimSize]
trainDf = trainDf[trainDf["Width"] >= dimSize]
trainDf = trainDf[trainDf["Width"].div(trainDf["Height"]) < 1.1]
trainDf = trainDf[trainDf["Height"].div(trainDf["Width"]) < 1.1]
trainDf = trainDf[["Path", "ClassId"]]
trainDf.to_csv("TestLabels.csv", index=False)
