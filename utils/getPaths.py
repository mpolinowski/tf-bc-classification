import os

# Defining image types you want to allow
imageTypes = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def listImages(basePath, contains=None):
    return listFiles(basePath, validExts=imageTypes, contains=contains)


def listFiles(basePath, validExts=None, contains=None):
    for (baseDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            # Get all files in filename / ignore empty directories
            if contains is not None and filename.find(contains) == -1:
                continue

            # Extracting the file extension
            fileExt = filename[filename.rfind("."):].lower()

            # Only process files that are of imageTypes
            if validExts is None or fileExt.endswith(validExts):
                # Construct the path to the image and yield it
                imagePath = os.path.join(baseDir, filename)
                yield imagePath
