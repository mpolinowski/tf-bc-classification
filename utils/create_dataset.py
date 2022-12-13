import config
import getPaths
import shutil
import random
import os


# Get content of the original input directory and shuffle images
allImagePaths = list(getPaths.listImages(config.RAW_INPUT_PATH))
random.seed(42)
random.shuffle(allImagePaths)

# Only take 10% of the images to speed up the process
print(len(allImagePaths))
imagePaths = allImagePaths[0:20000]
print(len(allImagePaths))

# Split into training and testing data
i = int(len(allImagePaths) * config.TRAIN_SPLIT)
trainPaths = allImagePaths[:i]
testPaths = allImagePaths[i:]

# Separate validation split from training data
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# Defining the datasets which will be build in the result folder
datasets = [
    ("training", config.TRAIN_PATH, trainPaths),
    ("validation", config.VAL_PATH, valPaths),
    ("testing", config.TEST_PATH, testPaths)
]

# Copy images from the initial into the result path
# while splitting them into train, validation and test data
for (dSType, basePath, allImagePaths) in datasets:
    print("Making '{}' split".format(dSType))
    if not os.path.exists(basePath):
        print("'Creating {}' directory".format(basePath))
        os.makedirs(basePath)
    # Looping over the image paths
    for inputPath in allImagePaths:
        # Extracting the filename of the input image
        # Extracting class label ("0" for "Benign" and "1" for "Malignant")
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        # Making the path to form the label directory
        labelPath = os.path.sep.join([basePath, label])
        if not os.path.exists(labelPath):
            print("'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        # Creating the path to the destination image and then copying it
        finalPath = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, finalPath)