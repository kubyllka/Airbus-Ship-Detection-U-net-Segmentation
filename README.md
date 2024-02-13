# Airbus-Ship-Detection-U-net-Segmentation
## Plan
* Find size of images
* Decode pixels function
* Encode pixels function
* Vizualize masks
* Find corrupted images
* Explore data
* Combine masks for every image
* Split for train and val set
* Create model
* Add augumentation
* Model learning
* Model testing
* Hyperparameter tuning

## Overview
The project was created to recognize ships in images (semantic segmentation). The dataset from the kaggle website was used for implementation. A slightly simplified U-net model was chosen as a model. Due to the lack of time and resources, training was conducted only on a part of the data, taking into account the balance of classes in the data. Augmentation and early stopping are used. Dice loss is used as a loss function. Adam optimizer was also used. Best perfomance: batch_size = 8, 20 epochs (but has early stopping), learning_rate=0.001, num_images_train_val  = 1000, percentage_img_with_ships = 0.7, percentage_val_split = 0.2

For faster training or reducing the load on the computer, it is recommended to use GPU or/and reduce the complexity of the model, or/and reduce the amount of data to be trained, or/and reduce batch_size.

## Main file
Understanding the initial data, searching for damaged files, merging data, creating a model and selecting hyperparameters, displaying the results in 

**kaggle-notebook.ipynb**


## Downloading the data

Go to [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview) page, log in and dowload the data (approximately 35 gigabytes). Also download this project.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

Go to folder with project -> Click on the Address Bar -> Type "cmd"

```bash
pip install -r requirements.txt
```

## Input .csv files
The input file for training the model must have two attributes **'ImageId'**, where the name of the photo is recorded, for example, '343efeee.jpg' and the attribute **'EncodedPixels'**, which contains enсoded (Run-length encoding, RLE) pixels that can be attributed to the ship.

The input file for testing the model must also have two attributes **'ImageId'**, **'EncodedPixels'** ('EncodedPixels' may be empty or has none values)

## Training model

To train the model, use the **train.py** file where you can set the parameters for training via cmd. More information about the parameters can be found using the command:
```bash
python train.py --help
```

For example, correct path to file:
```bash
D:\airbus-ship-detection\sample_submission_v2.csv
```

For example, correct path to images:
```bash
D:\airbus-ship-detection\train_v2\
```

The best model created during the training process will be saved.

Using example:
```bash
python train.py D:\airbus-ship-detection\train_ship_segmentations_v2.csv D:\airbus-ship-detection\train_v2\ --epochs 8 --batch_size 2 --learning_rate 0.01
```

## Testing model

To test the model, use the **inference.py** file where you can set the parameters for testing via cmd. More information about the parameters can be found using the command:

```bash
python inference.py --help
```

Using example:
```bash
python inference.py D:\airbus-ship-detection\sample_submission_v2.csv D:\airbus-ship-detection\test_v2\ model.h5
```

## Model
There is also an experimental model called **model.h5** that can be used for testing



