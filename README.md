# Breast Histopathology Image Segmentation

Using Tensorflow/Keras for Image Classifications


* [Part 1: Data Inspection and Pre-processing](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-10-tf-breast-cancer-classification-part1/2022-12-10)
* [Part 2: Weights, Data Augmentations and Generators](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-11-tf-breast-cancer-classification-part2/2022-12-11)
* [Part 3: Model creation based on a pre-trained and a custom model](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-11-tf-breast-cancer-classification-part3/2022-12-11)
* [Part 4: Train our model to fit the dataset](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-11-tf-breast-cancer-classification-part4/2022-12-11)
* [Part 5: Evaluate the performance of your trained model](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-12-tf-breast-cancer-classification-part5/2022-12-12)
* [Part 6: Running Predictions](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2022-12-12-tf-breast-cancer-classification-part6/2022-12-12)


> Based on [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) by [Paul Mooney](https://www.kaggle.com/paultimothymooney).
> `Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide.`
> [Can recurring breast cancer be spotted with AI tech? - BBC News](https://youtu.be/8XsiMQQ-4mM)

* Citation: [Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases](https://pubmed.ncbi.nlm.nih.gov/27563488/)
* Dataset: 198,738 IDC(negative) image patches; 78,786 IDC(positive) image patches