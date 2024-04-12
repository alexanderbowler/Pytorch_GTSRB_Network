import os
import torch
from NeuralNetworkClass import NeuralNetwork
from PIL import Image
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)

modelDir = "models/"
modelName = "DNN_model.pth"

if(os.path.isfile(modelDir+modelName)):
    model.load_state_dict(torch.load(modelDir+modelName))
    print("Loaded Model")
print(model)

CNNModel = NeuralNetwork().to(device)
if(os.path.isfile(modelDir+modelName)):
    CNNModel.load_state_dict(torch.load(modelDir+"CNN_model.pth"))
    print("Loaded Model")
print(CNNModel)



DataDir = "/home/albowler/projects/traffic_signs_project"
imgPath:str
while(True):
    print("Select an image from test dataset (ex. 00000.png  01207.png  02405.png  03487.png  04625.png  05762.png  06940.png  08056.png\n  09258.png  10407.png  11522.png "+ 
    "00002.png  01210.png  02407.png  03488.png  04626.png  05763.png  06942.png  08057.png  \n09261.png  10409.png  11523.png" + 
    "00004.png  01211.png  02409.png  03489.png  04628.png  05764.png  06944.png  08060.png  \n09262.png  10412.png  11531.png)\n")

    chosenImage = input()
    print("Chosen Image is: ", chosenImage)

    imgPath = os.path.join(DataDir, "curTest", chosenImage)
    if( not os.path.isfile(imgPath)):
        print("Error test file does not exist.")
    else:
        break

im = Image.open(imgPath)
im.show()

imgTensor = read_image(imgPath).type(torch.float32)
imgTensor = torch.reshape(imgTensor, (1, *imgTensor.shape))
imgTensor = imgTensor.to(device)
#plt.imshow(imgTensor.permute(1,2,0))
imgOneHot = model(imgTensor)
encodingToLabels = ["20 MPH", "30 MPH", "50 MPH", "60 MPH", "70 MPH", "80 MPH", "End 80 MPH",
                    "100 MPH", "120 MPH", "No Passing Zone", "No Passing Zone For Trucks", 
                   "Priority Road Sign",  "Preference Road Sign", "Yield", "Stop", "No cars", 
                   "No Trucks", "No Vehicles Any Kind", "Warning", "Left Curve", "Right Curve", 
                   "Left then Right Turn", "Rough Road", "Slipper When Wet", "Narrow Road", 
                   "Road Work Ahead", "Traffic Light Ahead", "Pedestrain", "Children Crossing",
                   "Bike Crossing", "Snow Warning", "Deer Crossing", "No Speed Limit", "Traffic Must Turn Right",
                   "Traffic Must Turn Left", "Traffic Must Go Straight", "Traffic Must Go Straight or Right",
                   "Traffic Must Go Straight or Left", "Traffic Keep Right", "Traffic Keep Left", "Round a bout",
                   "End of No Passing Zone", "End of No Truck Passing Zone"]
print("Predicted class: ", encodingToLabels[imgOneHot.argmax(1)[0].item()])