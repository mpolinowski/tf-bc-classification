from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import glob
import numpy as np
# from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve, auc, roc_auc_score
import config

# Defining the paths to the training, validation, and testing directories
trainPath = config.TRAIN_PATH
valPath = config.VAL_PATH
testPath = config.TEST_PATH

# Defining a method to get the number of files given a path
def retrieveNumberOfFiles(path):
    list1 = []
    for file_name in glob.iglob(path+'/**/*.png', recursive=True):
        list1.append(file_name)
    return len(list1)

# Checking for the total number of image paths in training, validation and testing directories
totalTrain = retrieveNumberOfFiles(config.TRAIN_PATH)
totalVal = retrieveNumberOfFiles(config.VAL_PATH)
totalTest = retrieveNumberOfFiles(config.TEST_PATH)

# Initialize the validation data augmentation object
valAug = ImageDataGenerator(rescale=1/255.0)

# Initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

# Loading the best performing model
# Please specify the model name from the output folder which has the lowest val_loss

# ResNet50
path1 = config.OUTPUT_PATH + '/resnet50_weights-025-0.6333.hdf5'

# Custom
# path1 = config.OUTPUT_PATH + '/custom_weights-009-0.4244.hdf5'

fModel = load_model(path1)

# Predicting on the test data
print("Predicting on the test data")
# if totalTest is odd number add `+1` to predTest
print("totalTrain: " , totalTrain , ", totalVal: " , totalVal , ", totalTest: " , totalTest)
# totalTrain:  199818 , totalVal:  22201 , totalTest:  55505
predTest = fModel.predict(x=testGen, steps=(totalTest // config.BATCH_SIZE)+1)
predTest = np.argmax(predTest, axis=1)

# Printing the Classification Report
print(classification_report(testGen.classes, predTest, target_names=testGen.class_indices.keys()))


# Computing the confusion matrix and and using the same to derive the 
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predTest)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# Printing the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))