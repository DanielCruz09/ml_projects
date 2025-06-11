"""This project seeks to evaluate five neural network architectures to predict if a weapon exists in an image."""

from collections import Counter
import xml.etree.ElementTree as ET
import os
import math
import pandas as pd
from skimage.transform import resize
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import models, layers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from plotnine import *

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

    # header_missing = False
    # with open(output_path, 'r') as csvfile:
    #     # check if the file has a header
    #     first_line = csvfile.readline()
    #     header_missing = not first_line.strip() 
    
        
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

def write_results(output_path:str, test_acc:float, val_acc:float, neurons:int, k:int, lr:float) -> None:
    """
    
    Write the base model results to a CSV.

    Parameters:
        output_path (str): CSV file to write results.
        test_acc (float): The testing accuracy for the base model.
        val_acc (float): The validation accuracy for the base model.
        neurons (int): Number of neurons used for the dense layer.
        k (int): The size of the convolution window.
        lr (float): The learning rate used to train the base model.

    Returns:
        None
    
    """
    header_missing = False
    with open(output_path, 'r') as csvfile:
        # check if the file has a header
        first_line = csvfile.readline()
        header_missing = not first_line.strip() 

    if header_missing:
        with open(output_path, 'w') as csvfile:
            header = 'Test Accuracy,Validation Accuracy,Neurons,K,Rate\n'
            csvfile.write(header)

    with open(output_path, 'a') as csvfile:
        line = f'{test_acc},{val_acc},{neurons},{k},{lr}\n'
        csvfile.write(line)

def create_cnn(n_neurons:int, input_shape:list[float], k:int, stride:int) -> tf.keras.models.Sequential:
    """
    
    Creates a Convolutional Neural Network. Primarily used for debugging.

    Parameters:
        n_neurons (int): Number of neurons to use in the dense layer.
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        k (int): The size of the convolution window.
        stride (int): Length of the convolution.

    Returns:
        model (tensorflow.keras.models.Sequential): The base model.
    
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, kernel_size=(k,k), strides=(stride,stride), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, kernel_size=(k,k), strides=(stride,stride), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, kernel_size=(k,k), strides=(stride,stride), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_neurons, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def train_network(model:tf.keras.models, training_X:np.ndarray, training_y:np.ndarray, lr:float, seed:int, log_to_csv:bool=False) -> tf.keras.callbacks.History:
    """
    
    Trains a TensorFlow.Keras model.

    Parameters:
        model (tf.keras.models): Model to train.
        training_X (numpy.ndarray): Set containing the training images.
        training_y (numpy.ndarray): Set containing the training labels.
        lr (float): The learning rate to use to train the model.
        seed (int): Random seed to use for reproducability.
        log_to_csv (bool): Write to a CSV if True.

    Returns:
        history (tf.keras.callbacks.History): A record of training loss values and metrics values at successive epochs, 
        as well as validation loss values and validation metrics values (if applicable).
    
    """
    # create validation sets
    train_X, train_y, valid_X, valid_y = create_validation(training_X, training_y, 0.20, seed)
    # Stop training when validation accuracy stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    # Record training
    csv_logger = tf.keras.callbacks.CSVLogger(filename='results.csv', separator=',')
    # Provide args to model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(), measures=['accuracy'])
    # Train the model
    if log_to_csv:
        # Include csv logger if the user specifies true
        history = model.fit(train_X, train_y, epochs=100, validation_data=(valid_X, valid_y), callbacks=[early_stopping, csv_logger], verbose=0)
        return history
    history = model.fit(train_X, train_y, epochs=100, validation_data=(valid_X, valid_y), callbacks=[early_stopping], verbose=0)
    return history

def hyperparameter_search(neurons:list[int], k_values:list[int], learning_rate:list[float], input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Performs a hyperparameter grid search. Writes the results to a CSV file.

    Parameters:
        neurons (list[int]): List containing number of neurons to use in the dense layer.
        k_values (list[int]): List containing sizes of the convolution window.
        learning_rate (list[float]): List containing the learning rates for training the model.
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    output_path = 'results.csv'
    for n in neurons:
        for k in k_values:
            for lr in learning_rate:
                model = create_cnn(n, input_shape, k, stride=1)
                history = train_network(model, train_X, train_y, lr, seed)
                test_acc = model.evaluate(test_X, test_y)[1]
                val_accuracies = history.history['val_accuracy']
                val_acc = np.mean(val_accuracies)
                write_results(output_path, test_acc, val_acc, n, k, lr)

def train_vggnet(input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Creates and trains a VGGNet model architecture. Writes the accuracy results to a CSV file.

    Parameters:
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    vgg_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='max',
        name='vgg16'
    )
    x = vgg_model.output
    out = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=vgg_model.input, outputs=out)
    print(model.summary())
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_network(model, train_X, train_y, lr, seed)
        test_acc = model.evaluate(test_X, test_y)[1]
        val_accuracies = history.history['val_accuracy']
        val_acc = np.mean(val_accuracies)
        print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')
        with open('model_results_v2.csv', 'a') as csvfile:
            line = f'VGGNet,{test_acc},{val_acc},{lr}\n'
            csvfile.write(line)

def train_xception(input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Creates and trains an Xception model architecture. Writes the accuracy results to a CSV file.

    Parameters:
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    xception_model = tf.keras.applications.Xception(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='max',
        name='xception'
    )
    x = xception_model.output
    out = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=xception_model.input, outputs=out)
    print(model.summary())
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_network(model, train_X, train_y, lr, seed)
        test_acc = model.evaluate(test_X, test_y)[1]
        val_accuracies = history.history['val_accuracy']
        val_acc = np.mean(val_accuracies)
        print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')
        with open('model_results_v2.csv', 'a') as csvfile:
            line = f'Xception,{test_acc},{val_acc},{lr}\n'
            csvfile.write(line)

def train_resnet(input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Creates and trains a ResNet50 model architecture. Writes the accuracy results to a CSV file.

    Parameters:
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    resnet_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='max',
        name='resnet50'
    )
    x = resnet_model.output
    # flatten = layers.Flatten()(x)
    dense = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=resnet_model.input, outputs=out)
    print(model.summary())
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_network(model, train_X, train_y, lr, seed)
        test_acc = model.evaluate(test_X, test_y)[1]
        val_accuracies = history.history['val_accuracy']
        val_acc = np.mean(val_accuracies)
        print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')
        with open('model_results_v2.csv', 'a') as csvfile:
            line = f'ResNet50,{test_acc},{val_acc},{lr}\n'
            csvfile.write(line)

# See https://dev.to/zohebabai/zfnet-ilsvrc-runner-up-2013-4hnj 
def train_zfnet(input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Creates and trains a ZFNet model architecture. Writes the accuracy results to a CSV file.

    Parameters:
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    zfnet_model = models.Sequential()
    zfnet_model.add(layers.Input(shape=input_shape))
    zfnet_model.add(layers.Conv2D(96, kernel_size=(7,7), strides=(2,2), activation='relu', padding='same'))
    zfnet_model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    zfnet_model.add(layers.Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', padding='same'))
    zfnet_model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    zfnet_model.add(layers.Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    zfnet_model.add(layers.Conv2D(384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    zfnet_model.add(layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    zfnet_model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    zfnet_model.add(layers.Flatten())
    zfnet_model.add(layers.Dense(4096, activation='relu'))
    zfnet_model.add(layers.Dense(4096, activation='relu'))
    zfnet_model.add(layers.Dense(1, activation='sigmoid'))

    print(zfnet_model.summary())
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_network(zfnet_model, train_X, train_y, lr, seed)
        test_acc = zfnet_model.evaluate(test_X, test_y)[1]
        val_accuracies = history.history['val_accuracy']
        val_acc = np.mean(val_accuracies)
        print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')
        with open('model_results_v2.csv', 'a') as csvfile:
            line = f'ZFNet,{test_acc},{val_acc},{lr}\n'
            csvfile.write(line)

# See https://www.geeksforgeeks.org/ml-getting-started-with-alexnet/ 
def train_alexnet(input_shape:list[float], train_X:np.ndarray, train_y:np.ndarray, test_X:np.ndarray, test_y:np.ndarray, seed:int) -> None:
    """
    
    Creates and trains an AlexNet model architecture. Writes the accuracy results to a CSV file.

    Parameters:
        input_shape (list[float]): Tuple or list containing 3 elements - the width, height, and channels.
        train_X (numpy.ndarray): Set containing the training images.
        train_y (numpy.ndarray): Set containing the training labels.
        test_X (numpy.ndarray): Set containing the testing images.
        test_y (numpy.ndarray): Set containing the testing labels.
        seed (int): Random seed to use for reproducability.

    Returns:
        None
    
    """
    model = models.Sequential() 
    model.add(layers.Resizing(224, 224, interpolation="bilinear", input_shape=input_shape))
    # 1st Convolutional Layer 
    model.add(layers.Conv2D(filters = 96, input_shape = input_shape, 
                kernel_size = (11, 11), strides = (4, 4), 
                padding = 'valid', activation='relu')) 
    # Max-Pooling 
    model.add(layers.MaxPooling2D(pool_size = (2, 2), 
                strides = (2, 2), padding = 'valid')) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # 2nd Convolutional Layer 
    model.add(layers.Conv2D(filters = 256, kernel_size = (11, 11), 
                strides = (1, 1), padding = 'valid', activation='relu')) 
    # Max-Pooling 
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                padding = 'valid')) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # 3rd Convolutional Layer 
    model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3), 
                strides = (1, 1), padding = 'valid', activation='relu'))   
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # 4th Convolutional Layer 
    model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3), 
                strides = (1, 1), padding = 'valid', activation='relu')) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # 5th Convolutional Layer 
    model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), 
                strides = (1, 1), padding = 'valid', activation='relu'))  
    # Max-Pooling 
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                padding = 'valid')) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # Flattening 
    model.add(layers.Flatten()) 
    # 1st Dense Layer 
    model.add(layers.Dense(4096, input_shape = input_shape, activation='relu')) 
    # Add Dropout to prevent overfitting 
    model.add(layers.Dropout(0.4)) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # 2nd Dense Layer 
    model.add(layers.Dense(4096, activation='relu')) 
    # Add Dropout 
    model.add(layers.Dropout(0.4)) 
    # Batch Normalisation 
    model.add(layers.BatchNormalization()) 
    # Output Softmax Layer 
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_network(model, train_X, train_y, lr, seed)
        test_acc = model.evaluate(test_X, test_y)[1]
        val_accuracies = history.history['val_accuracy']
        val_acc = np.mean(val_accuracies)
        print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')
        with open('model_results_v2.csv', 'a') as csvfile:
            line = f'AlexNet,{test_acc},{val_acc},{lr}\n'
            csvfile.write(line)

def plot_results(results:pd.DataFrame, output_path:str) -> None:
    """
    
    Plots a line chart of the accuracy results of each model.

    Parameters:
        results (pandas.DataFrame): Accuracy results of all models.
        output_path (str): Filepath to save the line chart.

    Returns:
        None
    
    """
    line_chart = (
        ggplot(results)
        + aes(x='Rate', y='Testing', color='Model')
        + geom_line()
        + ylim(0, 1)
        + ggtitle('Test Accuracies for Different Models')
    )
    line_chart.save(filename=output_path)

def calculate_conf_interval(measure:float, n:int) -> list[int]:
    """
    
    Calculates the confidence interval (with Bonferroni correction) for the testing accuracy.

    Parameters:
        measure (float): The testing accuracy measure.
        n (int): The length of the data set.

    Returns:
        interval (list[int]): Contains the lower and upper bound.
    
    """
    s = math.sqrt(measure * (1 - measure))
    se = s / math.sqrt(n)
    lower_bound = measure - (2.807 * se)
    upper_bound = measure + (2.807 * se)
    interval = [lower_bound, upper_bound]
    return interval

def plot_conf_interval(data:pd.DataFrame, n:int) -> None:
    """
    
    Plots the confidence interval for each model.

    Parameters:
        data (pandas.DataFrame): The dataset containing the models and accuracy results.
        n (int): The length of the data set.

    Returns:
        None
    
    """
    max_values = data.groupby('Model')['Testing'].max()
    lower = []
    upper = []
    for x in max_values:
        lower.append(calculate_conf_interval(x, n)[0])
        upper.append(calculate_conf_interval(x, n)[1])
    df = {'Model': ['VGGNet', 'Xception', 'ResNet50', 'ZFNet', 'AlexNet'], 'Best_Accuracy':max_values}
    df = pd.DataFrame(df)
    error_bar = (
        ggplot(df)
        + aes(x='Model', y='Best_Accuracy')
        + geom_point()
        + geom_errorbar(aes(ymin=lower, ymax=upper))
    )
    error_bar.save(filename='model_ci.png')

def get_label_proportion(data:np.array, output_path:str) -> None:
    """
    
    Generates a bar plot of the proportion of each label in the data.

    Parameters:
        data (numpy.array): Array of labels.
        output_path (str): Filepath to store the bar plot.
    
    """
    props = Counter(data)
    for key in props.keys():
        props[key] = props[key] / len(data)
    df = {'label':props.keys(), 'proportion':props.values()}
    df = pd.DataFrame(df)
    bar_chart = (
        ggplot(df)
        + aes(x='label', y='proportion')
        + geom_col()
        + ggtitle('Proportion of Labels')
    )
    bar_chart.save(filename=output_path)

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
    weapons_data = pd.read_csv('datasets/weapons_test.csv')
    min_width = weapons_data['Width'].min()
    min_height = weapons_data['Height'].min()
    images_path = 'datasets/Sohas_weapon-Detection/images_test/'
    
    img, labs = resize_images(images_path, min_width, min_height)

    resized_images = np.array(img)
    labels = np.array(labs)

    training_X, training_y, testing_X, testing_y = shuffle_split_data(resized_images, labels, 72, 0.80)

    # Input shape of (x, y, color channel)
    input_shape = (min_width, min_height, 3)
    
    with open('model_results_v2.csv', 'w') as csvfile:
        header = 'Model,Testing,Validation,Rate\n'
        csvfile.write(header)
    train_vggnet(input_shape, training_X, training_y, testing_X, testing_y, 72)
    train_xception(input_shape, training_X, training_y, testing_X, testing_y, 72)
    train_resnet(input_shape, training_X, training_y, testing_X, testing_y, 72)
    train_zfnet(input_shape, training_X, training_y, testing_X, testing_y, 72)
    train_alexnet(input_shape, training_X, training_y, testing_X, testing_y, 72)

    results = pd.read_csv('model_results_v2.csv')
    plot_results(results, 'model_results_line.png')
    plot_conf_interval(results, len(testing_X))
    get_label_proportion(labels, 'label_proportion.png')

if __name__ == "__main__":
    main()