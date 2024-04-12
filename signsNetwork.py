import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from trafficSignsDataset import ToTensor, TrafficSignsDataset
from NeuralNetworkClass import NeuralNetwork, CNN

batch_size = 64

#This is because data is stored in other folder and don't want to copy it over
DataDir = "/home/albowler/projects/traffic_signs_project"
training_data = TrafficSignsDataset(os.path.join(DataDir, "TrainLabels.csv"), DataDir )

test_data = TrafficSignsDataset(os.path.join(DataDir, "TestLabels.csv"), 
                                    DataDir)

# Create data loaders.
train_dataloader = DataLoader(training_data,  batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


if not torch.cuda.is_available():
    print("ERRORRR!!!!!")


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = CNN().to(device)

modelDir = "models/"
modelName = "CNN_model.pth"

if(os.path.isfile(modelDir+modelName)):
    model.load_state_dict(torch.load(modelDir+modelName))
    print("Loaded Model")
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 500
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    if t%10 == 0:
        print("Train Dataset Accuracy: \n")
        test(train_dataloader, model, loss_fn)
        print("Test Dataset Accuracy: \n")
        test(test_dataloader, model, loss_fn)
    if t%50 == 0:
        torch.save(model.state_dict(), modelDir+modelName)
        print("Saved PyTorch Model State to model.pth")
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



model.eval()
x, y = test_data[0][0], test_data[0][1]
for test_images, test_labels in train_dataloader:
    sample_img = test_images[0]
    sample_label = test_labels[0]
    break
print(sample_label)
print(test_images.size())
x = sample_img
x = torch.reshape(x, (1,3,45,45))
y = sample_label
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    print(pred, pred.size())
    predicted, actual = pred[0].argmax(0), y
    print(f'Predicted: "{predicted}", Actual: "{actual}"')