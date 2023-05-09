# CountPillar

CountPillar is a Python-based tool to generate synthetic datasets for counting and detecting pills. This repository creates synthetic images by adding pills to background images (e.g., a plate) to help develop and test machine learning models for pill counting and detection tasks.



## Setup the Environment

To install CountPillar, you will need to create a conda environment and install the required dependencie

### Installation
1. Create and activate a new conda environment:
    ```
    conda create -n countpillar python=3.10
    conda activate countpillar
    ```

2. Install all the dependencies using:
    ```
    pip install -r requirements.txt
    ```
3. Install the CountPillar package using:
    ```
    pip install -e .
    ```

### Development
To set up a development environment for CountPillar, you can use the following command:

    pip install -e .["dev"]

This will install additional development dependencies and allow you to make changes to the CountPillar code.

This repository uses pre-commits for `flake8`, `black`, and `isort`.
Run the following command from the root of the repository:

```shell script
pip install pre-commit
pre-commit install
```
to ensure that all code is properly linted & formatted before committing.

## Usage
### Generate Masks

To use CountPillar, you will first need to generate masks for the pill images in the `./data/pills/images` directory. This can be done by running the following command:
```
mask_generator
```
This will create mask images for each pill image and save them in the `data/pills/masks` directory.

### Generate Synthetic Dataset

After obtaining all the masks, you can create a synthetic dataset by running the following command:
```
dataset_generator
```
he generated dataset will be stored in the dataset/synthetic directory, with the following structure:
```
dataset/
    └── synthetic/
        ├── images/
        └── labels/
```

The `images` folder contains the synthetic images of pills composed on the background. The number of pills in each image is indicated in the filename. For example, a file named `0_18.jpg` implies that there are `18` pills in the image.

The `labels` folder contains bounding box annotations for each generated image, stored in the YOLO format. These annotations can be used for training and evaluating object detection models, specifically tailored for counting and detecting pills in images.
