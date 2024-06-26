import os
import torch
from torch import nn
from NeuralNetworkClass import NeuralNetwork, CNN
from PIL import Image
from torchvision.io import read_image
from trafficSignsDataset import ToTensor, TrafficSignsDataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import clip


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
#print(model)

CNNModel = CNN().to(device)
if(os.path.isfile(modelDir+modelName)):
    CNNModel.load_state_dict(torch.load(modelDir+"CNN_model.pth"))
    print("Loaded Model")
#print(CNNModel)



DataDir = "/home/albowler/projects/old_traffic_signs_project"
training_data = TrafficSignsDataset(os.path.join(DataDir, "TrainLabels.csv"), DataDir )

test_data = TrafficSignsDataset(os.path.join(DataDir, "TestLabels.csv"), 
                                    DataDir)

# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data,  batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
loss_fn = nn.CrossEntropyLoss()




encodingToLabels =  ["20 KPH German Traffic Sign", "30 KPH German Traffic Sign", "50 KPH German Traffic Sign", "60 KPH German Traffic Sign",
                     "70 KPH German Traffic Sign", "80 KPH German Traffic Sign", "End 80 KPH German Traffic Sign",
                    "100 KPH German Traffic Sign", "120 KPH German Traffic Sign", "No Passing Zone German Traffic Sign",
                      "No Passing Zone For Trucks German Traffic Sign", "Priority German Traffic Sign",  "Preference German Traffic Sign",
                    "Yield German Traffic Sign", "Stop German Traffic Sign", "No Cars Allowed German Traffic Sign", 
                   "No Trucks Allowed German Traffic Sign", "No Vehicles Any Kind German Traffic Sign", "Warning German Traffic Sign",
                     "Left Curve German Traffic Sign", "Right Curve German Traffic Sign", "Left then Right Turn German Traffic Sign",
                       "Rough Road German Traffic Sign", "Slipper When Wet German Traffic Sign", "Narrow Road German Traffic Sign", 
                   "Road Work Ahead German Traffic Sign", "Traffic Light Ahead German Traffic Sign", "Pedestrain German Traffic Sign",
                     "Children Crossing German Traffic Sign", "Bike Crossing German Traffic Sign", "Snow Warning German Traffic Sign",
                       "Deer Crossing German Traffic Sign", "No Speed Limit German Traffic Sign", "Traffic Must Turn Right German Traffic Sign",
                   "Traffic Must Turn Left German Traffic Sign", "Traffic Must Go Straight German Traffic Sign", 
                   "Traffic Must Go Straight or Right German Traffic Sign", "Traffic Must Go Straight or Left German Traffic Sign",
                     "Traffic Keep Right German Traffic Sign", "Traffic Keep Left German Traffic Sign", "Round a bout German Traffic Sign",
                   "End of No Passing Zone German Traffic Sign", "End of No Truck Passing Zone German Traffic Sign"]



# ["20 KPH German Traffic Sign", "30 KPH German Traffic Sign", "50 KPH German Traffic Sign", "60 KPH German Traffic Sign",
#                      "70 KPH German Traffic Sign", "80 KPH German Traffic Sign", "End 80 KPH German Traffic Sign",
#                     "100 KPH German Traffic Sign", "120 KPH German Traffic Sign", "No Passing Zone German Traffic Sign",
#                       "No Passing Zone For Trucks German Traffic Sign", "Priority German Traffic Sign",  "Preference German Traffic Sign",
#                     "Yield German Traffic Sign", "Stop German Traffic Sign", "No Cars Allowed German Traffic Sign", 
#                    "No Trucks Allowed German Traffic Sign", "No Vehicles Any Kind German Traffic Sign", "Warning German Traffic Sign",
#                      "Left Curve German Traffic Sign", "Right Curve German Traffic Sign", "Left then Right Turn German Traffic Sign",
#                        "Rough Road German Traffic Sign", "Slipper When Wet German Traffic Sign", "Narrow Road German Traffic Sign", 
#                    "Road Work Ahead German Traffic Sign", "Traffic Light Ahead German Traffic Sign", "Pedestrain German Traffic Sign",
#                      "Children Crossing German Traffic Sign", "Bike Crossing German Traffic Sign", "Snow Warning German Traffic Sign",
#                        "Deer Crossing German Traffic Sign", "No Speed Limit German Traffic Sign", "Traffic Must Turn Right German Traffic Sign",
#                    "Traffic Must Turn Left German Traffic Sign", "Traffic Must Go Straight German Traffic Sign", 
#                    "Traffic Must Go Straight or Right German Traffic Sign", "Traffic Must Go Straight or Left German Traffic Sign",
#                      "Traffic Keep Right German Traffic Sign", "Traffic Keep Left German Traffic Sign", "Round a bout German Traffic Sign",
#                    "End of No Passing Zone German Traffic Sign", "End of No Truck Passing Zone German Traffic Sign"]
# ["20 MPH", "30 MPH", "50 MPH", "60 MPH", "70 MPH", "80 MPH", "End 80 MPH",
#                     "100 MPH", "120 MPH", "No Passing Zone", "No Passing Zone For Trucks", 
#                    "Priority Road Sign",  "Preference Road Sign", "Yield", "Stop", "No cars", 
#                    "No Trucks", "No Vehicles Any Kind", "Warning", "Left Curve", "Right Curve", 
#                    "Left then Right Turn", "Rough Road", "Slipper When Wet", "Narrow Road", 
#                    "Road Work Ahead", "Traffic Light Ahead", "Pedestrain", "Children Crossing",
#                    "Bike Crossing", "Snow Warning", "Deer Crossing", "No Speed Limit", "Traffic Must Turn Right",
#                    "Traffic Must Turn Left", "Traffic Must Go Straight", "Traffic Must Go Straight or Right",
#                    "Traffic Must Go Straight or Left", "Traffic Keep Right", "Traffic Keep Left", "Round a bout",
#                    "End of No Passing Zone", "End of No Truck Passing Zone"]


#


def test(dataloader, len_dataset, model, loss_fn, ret_y_pred = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    if ret_y_pred:
        y_pred = torch.empty((0, 43), device = device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if(ret_y_pred):
                y_pred = torch.cat((y_pred, pred), 0)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if(ret_y_pred):
        return y_pred
    
def CLIP_test():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device = device)
    label_tokens = clip.tokenize(encodingToLabels).to(device)
    test_csv = pd.read_csv(os.path.join(DataDir, 'TestLabels.csv'))
    processed_images = []

    for row in range(test_csv.shape[0]):
        processed_images.append(clip_preprocess(Image.open(os.path.join(DataDir,test_csv.iloc[row,0]))))
    y_true = torch.tensor(test_csv.iloc[:,1]).to(device)

    processed_images = torch.stack(processed_images).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(processed_images, label_tokens)
        y_pred = logits_per_image.argmax(dim = -1)
        
        # print(y_pred[:10])
        # print(y_true[:10])
        y_match = (y_pred == y_true).type(torch.float).mean()
    print(f"CLIP model Test Error: \n Accuracy: {y_match.item():>0.1f}%\n")

CLIP_test()


print("DNN Train Dataset Accuracy: ")
test(train_dataloader, len(training_data), model, loss_fn)
print("DNN Test Dataset Accuracy: ")
dnn_y_test_pred = test(test_dataloader, len(test_data), model, loss_fn, ret_y_pred = True)

print("CNN Train Dataset Accuracy: ")
test(train_dataloader, len(training_data), CNNModel, loss_fn)
print("CNN Test Dataset Accuracy: ")
cnn_y_test_pred  = test(test_dataloader, len(test_data), CNNModel, loss_fn, ret_y_pred = True)


y_test_true = torch.tensor(pd.read_csv(DataDir+'/TestLabels.csv')["ClassId"].values)
#print(y_test_true[0:100])
dnn_y_test_pred = torch.argmax(dnn_y_test_pred, dim=1)
cnn_y_test_pred = torch.argmax(cnn_y_test_pred, dim=1)
#print(dnn_y_test_pred[0:100])

cf_matrix = confusion_matrix(y_test_true.cpu(), dnn_y_test_pred.cpu())
#print(cf_matrix)
# for i in range(len(cf_matrix)):
#     cf_matrix[i][i] = 1 - cf_matrix[i][i]

#print(cf_matrix)
dnn_cm_df = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in encodingToLabels],
                     columns = [i for i in encodingToLabels])

cf_matrix = confusion_matrix(y_test_true.cpu(), cnn_y_test_pred.cpu())
cnn_cm_df = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in encodingToLabels],
                     columns = [i for i in encodingToLabels])


plt.figure(figsize = (40,20))
sb.heatmap(dnn_cm_df, annot=True)
plt.savefig('dnn_confusion_matrix.png')

plt.figure(figsize = (40,20))
sb.heatmap(cnn_cm_df, annot=True)
plt.savefig('cnn_confusion_matrix.png')


imgPath:str
while(True):
    print("Select an image from test dataset (ex. 00252.png 00000.png  01207.png  02405.png  03487.png  04625.png  05762.png  06940.png  08056.png\n  09258.png  10407.png  11522.png "+ 
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
imgOneHotCnn = CNNModel(imgTensor)

print("DNN Predicted class: ", encodingToLabels[imgOneHot.argmax(1)[0].item()])
print("CNN Predicted class: ", encodingToLabels[imgOneHotCnn.argmax(1)[0].item()])
