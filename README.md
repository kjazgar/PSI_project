# Flowers recognition project

### Description
Artificial Intelligence project that aim was to classification images of flowers into five different kinds: daisy, dandelion, rose, sunflower and tuilp. 

### Dataset
The data collection is dowloaded from https://www.kaggle.com/alxmamaev/flowers-recognition, contains 4242 images of flowers and is based on the data flicr, google images, yandex images. The pictures are divided into five classes: daisy, dandelion, rose, sunflower and tuilp and there about 800 photos for each class.
### Overview
`read_data.py` script that allows to read the data and print some random images. 

`reprocess_data.py` script that allows to reprocess the data and split it into train, test and validation sets. 

`logistic.py` file with function that creates logistic regression with parameter C found using GridSearchCV.

`svm_linear.py` file contains function that creates SVM model with linear kernel and parameter C found using GridSearchCV.

`svm_poly.py` file contains function that creates SVM model with polynomial kernel and parameter C found using GridSearchCV.

`model8.py` file contains function that creates neural network model with following structure:
```
Model: "sequential"
Layer (type)                 Output Shape              Param #   
================================================================= 
conv2d (Conv2D)              (None, 128, 128, 64)      1792      
_________________________________________________________________
module_wrapper (ModuleWrappe (None, 64, 64, 64)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 64, 64)        256       
_________________________________________________________________
dropout (Dropout)            (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 128)       73856     
_________________________________________________________________
module_wrapper_1 (ModuleWrap (None, 32, 32, 128)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 128)       512       
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       147584    
_________________________________________________________________
module_wrapper_2 (ModuleWrap (None, 16, 16, 128)       0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512       
_________________________________________________________________
dropout_2 (Dropout)          (None, 16, 16, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 256)       295168    
_________________________________________________________________
module_wrapper_3 (ModuleWrap (None, 8, 8, 256)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 256)         1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 256)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 512)         1180160   
_________________________________________________________________
module_wrapper_4 (ModuleWrap (None, 4, 4, 512)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 512)         2048      
_________________________________________________________________
dropout_4 (Dropout)          (None, 4, 4, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              8389632   
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 1024)              4096      
=================================================================
```

`model10.py` file contains function that creates neural network model with following structure:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 16)      448       
_________________________________________________________________
module_wrapper (ModuleWrappe (None, 64, 64, 16)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 64, 16)        64        
_________________________________________________________________
dropout (Dropout)            (None, 64, 64, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        4640      
_________________________________________________________________
module_wrapper_1 (ModuleWrap (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 64)        18496     
_________________________________________________________________
module_wrapper_2 (ModuleWrap (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
module_wrapper_3 (ModuleWrap (None, 8, 8, 128)         0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 128)         147584    
_________________________________________________________________
module_wrapper_4 (ModuleWrap (None, 4, 4, 128)         0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 128)         512       
_________________________________________________________________
dropout_4 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
module_wrapper_5 (ModuleWrap (None, 2, 2, 256)         0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 2, 2, 256)         1024      
_________________________________________________________________
dropout_5 (Dropout)          (None, 2, 2, 256)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 2, 512)         1180160   
_________________________________________________________________
module_wrapper_6 (ModuleWrap (None, 1, 1, 512)         0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 1, 1, 512)         2048      
_________________________________________________________________
dropout_6 (Dropout)          (None, 1, 1, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 1024)              525312    
_________________________________________________________________
dropout_7 (Dropout)          (None, 1024)              0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 1024)              4096      
_________________________________________________________________
activation (Activation)      (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 5125      
=================================================================
```
### Project structure
```
.
├── README.md
├── model1.py
├── read_data.py
└── reprocess_data.py
```
