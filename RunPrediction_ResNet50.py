from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils import config
import numpy as np
import argparse
import cv2
import os

# Model you want to use
modelName = 'resnet50_weights.hdf5'
modelPath = config.OUTPUT_PATH + '/' + modelName

# Loading the Breast Cancer detector model
print("Loading Breast Cancer detector model...")
model = load_model(modelPath)


# Test image
# imagePath ="./sample_pictures/malignant.png"
imagePath =  "./sample_pictures/benign.png"

# Loading the input image using openCV
image = cv2.imread(imagePath)

# Convert it from BGR to RGB and then resize it to 48x48, 
# the same parameter we used for training
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = cv2.resize(image1, (48, 48))
image1 = img_to_array(image1)

# Only for ResNet50
image1 = preprocess_input(image1)

# The image is now represented by a NumPy array of shape (48, 48, 3), however 
# we need the dimensions to be (1, 3, 48, 48) so we can pass it
# through the network and then we'll also preprocess the image by subtracting the 
# mean RGB pixel intensity from the ImageNet dataset
image1 /=  255.0

image1 = np.expand_dims(image1, axis=0)

# Pass the image through the model to determine if the person has malignant
(benign, malignant) = model.predict(image1)[0]


# Adding the probability in the label
label = "benign" if benign > malignant else "malignant"
label = "{}: {:.2f}%".format(label, max(benign, malignant) * 100)
        
# Showing the output image
print("RESULT :" +label)

cv2.imshow("IDC", image)
if cv2.waitKey(5000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()