# Histopathologic Cancer Detection Kaggle Mini Project

From the original competition on Kaggle (https://www.kaggle.com/c/histopathologic-cancer-detection/overview) I will create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans.
Github Repo: https://github.com/GHeart01/CNN_Cancer_Detection
Contents
Description
Exploratory Data Analysis
DModel Architecture
Results and Analysis
Conclusion
#### Table of Contents

- [Description](#Description)
- [EDA](#Exploratory-Data-Analysis-(EDA)-Procedure)
- [Model Architecture](#Model-Architecture)
- [Result and Analysis](#Result-and-Analysis)
- [Conclusion](#Conclusion)
- [Citation](#Citation)
#### Description
In this Project, I create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset.

PCam is highly interesting for both its size, simplicity to get started on, and approachability. Determination of success is based on area under the ROC curve between predicted probablity and the observed target.
import subprocess # To avoid extremely long terminal output

packages = ["tensorflow", "tensorflow_decision_forests", "ydf", 
            "statsmodels","pandas","seaborn","numpy","matplotlib",
            "scipy",
            "scikit-learn"
           ]

#### Exploratory Data Analysis (EDA) Procedure
In this dataset I am provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. You are predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.

#### Result and Analysis
# Prediction probability histogram

The MobileNetV2 model achieve an excellent performance of .93 on the ROC curve after only 3 epoch and extremely reduced training data size. Based on the performance metric resport I see there were 40 false negatives and 74 false positives. The default threshhold of .5 has 61 false negative sand 61 false positives, which is more balanced but misses some positive cancer cases. 
This extremely strong performance after only 3 epochs with less than 10% of parameters being trained and an average precision of .9039 is incredible. Lowering the threshold slightly (from 0.5 to 0.401) improved the recall for Cancer cases — meaning the model caught more true positives, which is vital in medical diagnostics where missing a cancer case can be critical.
#### Conclusion
Despite this very strong performance, it is very clear that more epoch, more training data, training longer, augmenting data, and a more robust model or a custom CNN would perform better. My limiting factor here is a lack of computing power, and an unwillingness to wait for an hours to build a model.  
#### Citation
Will Cukierski. Histopathologic Cancer Detection. https://kaggle.com/competitions/histopathologic-cancer-detection, 2018. Kaggle.