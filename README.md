# GTSRB Classification with basic Convolutional and Deep Neural Networks
Ended up with 84.5% accuracy with the simple NN model and 88.2% accuracy with the simple CNN on a test set.

## Dataset
The GTSRB Dataset consists of a variety of german traffic signs. It consists of over 40 classes with over 50,000 images to train on. It has been a classic computer vision benchmark since it was released in 2011. Although by modern standards it is easily beaten I thought it was a fun dataset for an initial dive into image classification machine learning. A quick view of some of the various different roadsigns in this dataset:
![image](https://github.com/user-attachments/assets/29c4f050-b1b2-49e8-9001-68b57f6ae53e)

## Preprocessing
As a preprocessing step I resized all of the images that were over 45 pixels wide and tall, I also only utilized images which were within an aspect ratio of height/width < 1.1 and width/height < 1.1. I made this choice to prevent distorting the image too much. This resulted in just over 14k training images and ~4500 test images.

## Models 
### Simple Non convolutional Neural Network
The first model I made to attempt this dataset was a simple Neural Network with a singular hidden layer utilizing the ReLU activation function and normalizing the output with the logisitic softmax function. Achieved 84.5% test accuracy when trained to 100% train accuracy.

### Simple Convolutional Neural Network
The second model I made was a simple convolutional Neural Network with 2 convolutional layers with 8 and then 16 filters respectively both with 5x5 filters. After the convolutional layers was a Max Pooling Layer with 2x2 pool, and stride length 2. Then we appended the same Neural Network Architecture as earlier to the flattened output o f the convolutional layers. Achieving a final test accuracy of 88.2%.
