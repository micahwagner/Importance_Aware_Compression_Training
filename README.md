# Importance_Aware_Compression_Training

This project allows you to train on CIFAR10 / ImageNet100 using resnet18 / resnet50 on our importance aware compression algorithm. The dataset and model used can be switched on lines 34 and 35 in main.py respectively. By default, it's set to use CIFAR10 and resnet18. On line 37, you can choose whether you want the data to be pre-compressed or not. This means that instead of compressing the images on the fly, we generate X amount of compressed datasets of CIFAR10 (X corresponding to the optimal number of clusters according to the profiling stage).


To run on CIFAR10, first set the route to where you want CIFAR10 to be installed on line 23 in main.py. By default, it downloads in ./data/CIFAR10. Within the compress/generate_jpegs.py script, you can also change where the precompressed data is stored. By default, it's also set to ./data.


To run on ImageNet100, you first need to download the dataset from kaggle (https://www.kaggle.com/datasets/ambityga/imagenet100), and edit line 21 to point to where the dataset is. Next, you need to put all the contents of every train.x folder into one folder called "train", and rename the val.x folder to just "val". As of now, pre-compression only works with CIFAR10.


Lines 198 to 201 have configurable variables to set while running the importance aware compression. The variables are mode,fixed_quality, manual_thresholds, and fixed_test_quality. The three different modes are fixed (train on a fixed compression level, set fixed_quality accordingly), manual (set manual_thresholds for H/M/L compression categories), and cluster (run the dynamic importance aware compression). By defult it will just run on the cluster mode. You can also change the number of epochs on line 133.


To run the project, make sure to download the dependencies from the enviroment.yml in a conda environment. Once you have finished configuring the scripts, run "python main.py profile" to run the profiling stage, then simply "python main.py" to run a training session which will use our importance aware compression.
