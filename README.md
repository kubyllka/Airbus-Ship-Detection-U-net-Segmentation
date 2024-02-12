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

## Downloading the data

Go to [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview) page, log in and dowload the data (approximately 35 gigabytes)


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

Go to folder with project -> Click on the Address Bar -> Type "cmd"

```bash
pip install -r requirements.txt
```

For example, correct path to file:
```bash
D:\airbus-ship-detection\sample_submission_v2.csv
```

For example, correct path to images:
```bash
D:\airbus-ship-detection\train_v2\
```


## Training model

More info about parameters:
```bash
python train.py --help
```

## Testing model

More info about parameters:
```bash
python inference.py --help
```


