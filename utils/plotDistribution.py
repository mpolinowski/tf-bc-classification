# run `pipenv run python ./plotDistribution.py exploreData`
# change path in `exploreData` to print distributions

import sys
import glob
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploreData():
    plotData(config.TRAIN_PATH,"Training Path")
    # plotData(config.VAL_PATH,"Validation Path")
    # plotData(config.TEST_PATH,"Test Path")

# Plotting the count of images within each segment in a directories
def plotData(dirPath,graphTitle):
    # Get the path to the benign and malignant sub-directories
    benign_cases_dir = dirPath+ '/0/'
    malignant_cases_dir = dirPath + '/1/'

    # Get the list of all the images from those paths
    benign_cases = glob.glob(benign_cases_dir + '*.png')
    malignant_cases = glob.glob(malignant_cases_dir + '*.png')

    # An empty list
    data1 = []

    # Add all benign cases to list with label `0`
    for img in benign_cases:
        data1.append((img,0))

    # Add all benign cases to list with label `1`
    for img in malignant_cases:
        data1.append((img, 1))

    # data => pandas dataframe
    data1 = pd.DataFrame(data1, columns=['image', 'label'],index=None)

    # Shuffle the data 
    data1 = data1.sample(frac=1.).reset_index(drop=True)
    
    # Get the counts for each segment
    cases_count = data1['label'].value_counts()
    print(cases_count)

    # Plot the results 
    plt.figure(figsize=(10,8))
    sns.barplot(x=cases_count.index, y= cases_count.values)
    plt.title(graphTitle, fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['benign(0)', 'malignant(1)'])
    plt.show()

try:
    function = sys.argv[1]
    globals()[function]()
except IndexError:
    raise Exception("Please provide function name")
except KeyError:
    raise Exception("Function {} hasn't been found".format(function)) 
