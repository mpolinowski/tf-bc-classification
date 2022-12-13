from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.optimizers import Adam
from utils import config
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import seaborn as sns
import cv2



# Method to get the number of files given a path
def retrieveNumberOfFiles(path):
    list1 = []
    for file_name in glob.iglob(path+'/**/*.png', recursive=True):
        list1.append(file_name)
    return len(list1)


# Defining the paths to the training, validation, and testing directories
trainPath = config.TRAIN_PATH
valPath = config.VAL_PATH
testPath = config.TEST_PATH

# Checking for the total number of images
totalTrain = retrieveNumberOfFiles(config.TRAIN_PATH)
totalVal = retrieveNumberOfFiles(config.VAL_PATH)
totalTest = retrieveNumberOfFiles(config.TEST_PATH)


# Defining a method to get the list of files given a path
def getAllFiles(path):
    list1 = []
    for file_name in glob.iglob(path+'/**/*.png', recursive=True):
        list1.append(file_name)
    return list1

# Retrieving all files from train directory
allTrainFiles = getAllFiles(config.TRAIN_PATH)


# Calculating the total number of training images against each class and then store the class weights in a dictionary
trainLabels = [int(p.split(os.path.sep)[-2]) for p in allTrainFiles]
trainLabels = to_categorical(trainLabels)
classSumTotals = trainLabels.sum(axis=0)
classWeight = dict()

# Looping over all classes and calculate the class weights
for i in range(0, len(classSumTotals)):
    classWeight[i] = classSumTotals.max() / classSumTotals[i]


# Defining a method to plot training and validation accuracy and loss
# H - model fit
# N - number of training epochs
# plotPath - where to store the output file
def training_plot(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


# Initialize the training data augmentation object
## preprocess_input will scale input pixels between -1 and 1
## rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
## zoom_range is for randomly zooming inside pictures
## width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
## shear_range is for randomly applying shearing transformations
## horizontal_flip and vertical_flip is for randomly flipping half of the images horizontally and vertically resp
## fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift

trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# Initialize the training generator
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE)


# Initialize the validation data augmentation object
valAug = ImageDataGenerator(rescale=1/255.0)


# Initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

# Initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)


# Loading the ResNet50, ensuring the head Full Connected layers are left off / removed
baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(48, 48, 3)))

# Construct the head of the model that will be placed on top of the the base model
## Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. 
## We flatten the output of the convolutional layers to create a single long feature vector. 
### Average pooling computes the average of the elements present in the region of feature map covered by the filter.
#### ReLU stands for Rectified Linear Unit. 
#### The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time.
## Dropout is a technique where randomly selected neurons are ignored during training.

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7), padding="same")(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# Placing the head model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all the layers of the base model and freeze them so that they are 
# not updated during the training process
for layer in baseModel.layers:
    layer.trainable = False


# Compiling the model
## Decay updates the learning rate by a decreasing factor in each epoch
print("Compiling model")
opt = Adam(learning_rate=config.INIT_LR / config.EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# Using ModelCheckpoint to store the best performing model based on val_loss
# MCName = os.path.sep.join([config.OUTPUT_PATH, "resnet50_weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
# changed to static name to be able to pipe it into eval below
MCName = os.path.sep.join([config.OUTPUT_PATH, "resnet50_weights.hdf5"])
checkpoint = ModelCheckpoint(MCName, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]


# Fitting the model on training data
print("Model Fitting")
MF = model.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    class_weight=classWeight,
    callbacks=callbacks,
    epochs=config.EPOCHS)


# Evaluation

## load latest weights
path1 = config.OUTPUT_PATH + '/resnet50_weights.hdf5'

fModel = load_model(path1)

# Predicting on the test data
print("Predicting on the test data")
# if totalTest is odd number add `+1` to predTest
print("totalTrain: " , totalTrain , ", totalVal: " , totalVal , ", totalTest: " , totalTest)
# totalTrain:  199818 , totalVal:  22201 , totalTest:  55505
predTest = fModel.predict(x=testGen, steps=(totalTest // config.BATCH_SIZE)+1)
predTest = np.argmax(predTest, axis=1)

# Calculate roc auc
XGB_roc_value = roc_auc_score(testGen.classes, predTest)
print("XGboost roc_value: {0}" .format(XGB_roc_value))

# Plotting the graph
training_plot(MF, config.EPOCHS, config.PLOT_PATH_RN)


# # Serialize/Writing the model to disk
# print("Serializing network...")
# fModel.save(config.MODEL_PATH, save_format="h5")