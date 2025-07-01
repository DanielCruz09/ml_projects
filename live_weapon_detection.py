"""This project seeks to detect weapons in real-time footage."""

from collections import Counter
import xml.etree.ElementTree as ET
import os
import math
import pandas as pd
from skimage.transform import resize
import numpy as np
from PIL import Image
# import tensorflow as tf
# from keras import models, layers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from plotnine import *
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

# create a csv: filename, width, height, depth, xmin, ymin, xmax, ymax
text = ''
def write_to_csv(filepath:str, output_path:str) -> None:
    """

    Writes elements from an XML file to a CSV file.

    Parameters:
        filepath (str): Filepath pointing to the XML file
        output_path (str): Name of the CSV file to write

    Returns:
        None

    """
    tree = ET.parse(filepath)
    root = tree.getroot()
        
    # write contents
    text = ''
    for child in root:
        if child.tag == 'filename':
            text += str(child.text)
        if child.tag == 'size':
            # 0 = width, 1 = height, 2 = depth
            for i in range(3):
                text += ',' + str(child[i].text)
        if child.tag == 'object':
            # 0 = xmin, 1 = ymin, 2 = xmax, 3 = ymax
            for i in range(4):
                text += ',' + str(child[4][i].text)
            # There may be multiple bounding boxes
            # For now, only consider the first one
            break

    with open(output_path, 'a') as csvfile:
        csvfile.write(text)
        csvfile.write('\n')

def has_weapon(path:str) -> bool:
    """
    
    Detects if there is a weapon name in the filepath. Used to generate the labels.

    Parameters:
        path (str): The filepath to detect for weapons.

    Returns:
        bool: True if filepath contains a weapon. False, otherwise.
    
    """
    if ('knife' in path or 'Knife' in path or 'pistol' in path):
        return True
    return False

def resize_images(images_path:str, resized_width:int, resized_height:int) -> tuple:
    """
    
    Resizes images to a given width and height.

    Parameters:
        images_path (str): Filepath containing the images to resize.
        resized_width (int): Width to resize the images.
        resized_height (int): Height to resize the images.

    Returns:
        tuple: NumPy arrays containing the resized images and generated labels.
    
    """
    # Resize images
    resized_images = []
    labels = []
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        if os.path.isfile(file_path):
            img = Image.open(file_path)
            img_data = np.asarray(img)
            resized = resize(img_data, (resized_width, resized_height))
            # print(resized.shape)
            resized_images.append(resized)
            if has_weapon(file_path):
                labels.append(1)
            else:
                labels.append(0)
    return resized_images, labels

def shuffle_split_data(images:np.array, labels:np.array, seed:int, train_percentage:float) -> tuple:
    """
    
    Shuffles datasets into training and testing sets.

    Parameters:
        images (numpy.array): NumPy Array containing the images.
        labels (numpy.array): NumPy Array containing the labels.
        seed (int): Random seed to use for reproducability.
        train_percentage (float): Percentage to use for the training set.
    
    Returns:
        training_X (numpy.ndarray): Training image set.
        testing_X (numpy.ndarray): Testing image set.
        training_y (numpy.ndarray): Training label set.
        testing_y (numpy.ndarray): Testing label set.
    
    """
    # shuffle sets, keeping each image with its corresponding label
    X, y = shuffle(images, labels, random_state=seed)
    # create training and testing sets
    training_X, testing_X, training_y, testing_y = train_test_split(X, y, train_size=train_percentage, random_state=seed)

    return training_X, training_y, testing_X, testing_y

def create_validation(training_X:np.ndarray, training_y:np.ndarray, valid_percentage:float, seed:int) -> tuple:
    """
    
    Creates validation sets.

    Parameters:
        training_X (numpy.ndarray): Set containing the training images.
        training_y (numpy.ndarray): Set containing the training labels.
        valid_percentage (float): Percentage of data to use for validation.
        seed: Random seed to use for reproducability.

    Returns:
        train_X (numpy.ndarray): Training image set.
        valid_X (numpy.ndarray): Validation image set.
        train_y (numpy.ndarray): Training label set.
        valid_y (numpy.ndarray): Validation label set.
    
    """
    # create validation sets and smaller training sets
    train_X, valid_X, train_y, valid_y = train_test_split(training_X, training_y, test_size=valid_percentage, random_state=seed)

    return train_X, train_y, valid_X, valid_y


def train_model(training_X):
    '''Check https://docs.pytorch.org/vision/stable/models.html for source code.'''
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(training_X)]

    return weights, model, batch

def make_predictions(weights, img, model, batch):
    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                        labels=labels,
                        colors="red",
                        width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()


def main():

    write = False # Set this to true to write XML files to CSV
    if write:

        with open('weapons.csv', 'a') as csvfile:
            header = 'Filename,Width,Height,Depth,Xmin,Ymin,Xmax,Ymax'
            csvfile.write(header)
            csvfile.write('\n')

        with open('weapons_test.csv', 'a') as csvfile:
            header = 'Filename,Width,Height,Depth,Xmin,Ymin,Xmax,Ymax'
            csvfile.write(header)
            csvfile.write('\n')

        path = 'datasets/Sohas_weapon-Detection/annotations/xmls/'

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                write_to_csv(file_path, 'weapons.csv')

        test_path = 'datasets/Sohas_weapon-Detection/annotations_test/xmls/'

        for filename in os.listdir(test_path):
            file_path = os.path.join(test_path, filename)
            if os.path.isfile(file_path):
                write_to_csv(file_path, 'weapons_test.csv')

    # Get min dimensions
    weapons_data = pd.concat(
        map(pd.read_csv, ['datasets/weapons.csv', 'datasets/weapons_test.csv']), ignore_index=True
    )
    min_width = weapons_data['Width'].min()
    min_height = weapons_data['Height'].min()
    images_path = 'datasets/images_full'
    
    img, labs = resize_images(images_path, min_width, min_height)

    resized_images = np.array(img)
    labels = np.array(labs)

    training_X, training_y, testing_X, testing_y = shuffle_split_data(resized_images, labels, 72, 0.80)
    print(testing_X)

    # Input shape of (x, y, color channel)
    input_shape = (min_width, min_height, 3)

    weights, model, batch = train_model(training_X)
    img =  Image.fromarray(testing_X[0])
    make_predictions(weights, img, model, batch)


if __name__ == "__main__":
    main()