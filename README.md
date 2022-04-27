### Malaria Cell Infection Detection Model

This project implements a basic Tensorflow model using convolutional layers in order to classify cell images. The possible classes are healthy and infected. 

**Dataset URL:** 

https://storage.googleapis.com/open-ml-datasets/malaria-cells-dataset/cell_images.zip

### Instructions:

Simply run the main.py file and wait till it returns a zero code. The script downloads a dataset, preprocess it and then use it tp train a neural network model. Once that is ready, it saves the model in H5 format and the training results on a logs folder and reports folder. The logs folder can be accessed using the tensorboard UI. The reports are in PNG format. Finally, it makes predictions upon the test dataset and calculate the precision and recall metrics and print it on the console. The PNG report contains the data to calculate the metrics (Confusion Matrix). 

### Env Requirements:

Python 3.9\
Use pip install -r ./requirements.txt