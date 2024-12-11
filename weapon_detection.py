import xml.etree.ElementTree as ET
import os
import pandas as pd
from skimage.transform import resize
import numpy as np
from PIL import Image
from tensorflow.keras import models, layers
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# create a csv: filename, width, height, depth, xmin, ymin, xmax, ymax
def write_to_csv(filepath, output_path):
    
    tree = ET.parse(filepath)
    root = tree.getroot()

    header_missing = False
    with open(output_path, 'r') as csvfile:
        # check if the file has a header
        first_line = csvfile.readline()
        header_missing = not first_line.strip() 
        
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
        if header_missing:
            header = 'Filename,Width,Height,Depth,Xmin,Ymin,Xmax,Ymax'
            csvfile.write(header)
            csvfile.write('\n')
        csvfile.write(text)
        csvfile.write('\n')

def has_weapon(path):
    if ('knife' in path or 'Knife' in path or 'pistol' in path):
        return True
    return False

def resize_images(images_path, resized_width, resized_height):
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

def shuffle_split_data(images, labels, seed, train_percentage):

    # shuffle sets, keeping each image with its corresponding label
    X, y = shuffle(images, labels, random_state=seed)
    # create training and testing sets
    training_X, testing_X, training_y, testing_y = train_test_split(X, y, train_size=train_percentage, random_state=seed)

    return training_X, training_y, testing_X, testing_y

def create_validation(training_X, training_y, valid_percentage, seed):
    # create validation sets and smaller training sets
    train_X, valid_X, train_y, valid_y = train_test_split(training_X, training_y, test_size=valid_percentage, random_state=seed)

    return train_X, train_y, valid_X, valid_y

def write_results(output_path, test_acc, val_acc, neurons, k, lr):
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

def create_cnn(n_neurons, input_shape, k, stride):
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

def train_network(model, training_X, training_y, lr, seed, log_to_csv=False):
    # create validation sets
    train_X, train_y, valid_X, valid_y = create_validation(training_X, training_y, 0.20, seed)
    # Stop training when validation accuracy stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # Record training
    csv_logger = tf.keras.callbacks.CSVLogger(filename='results.csv', separator=',')
    # Provide args to model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # Train the model
    if log_to_csv:
        # Include csv logger if the user specifies true
        history = model.fit(train_X, train_y, epochs=100, validation_data=(valid_X, valid_y), callbacks=[early_stopping, csv_logger], verbose=0)
        return history
    history = model.fit(train_X, train_y, epochs=100, validation_data=(valid_X, valid_y), callbacks=[early_stopping], verbose=0)
    return history

def hyperparameter_search(neurons, k_values, learning_rate, input_shape, train_X, train_y, test_X, test_y, seed):
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

def train_vggnet(input_shape, train_X, train_y, test_X, test_y, lr, seed):
    vgg_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='max',
        name='vgg16'
    )
    x = vgg_model.output
    # flatten = layers.Flatten()(x)
    dense = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=vgg_model.input, outputs=out)
    print(model.summary())
    history = train_network(model, train_X, train_y, lr, seed)
    test_acc = model.evaluate(test_X, test_y)[1]
    val_accuracies = history.history['val_accuracy']
    val_acc = np.mean(val_accuracies)
    print(f'Test accuracy: {test_acc}, Validation Accuracy: {val_acc}')

def main():

    write = False # Set this to true to write XML files to CSV
    if write:
        path = 'Sohas_weapon-Detection/annotations/xmls/'

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                write_to_csv(file_path, 'weapons.csv')

        test_path = 'Sohas_weapon-Detection/annotations_test/xmls/'

        for filename in os.listdir(test_path):
            file_path = os.path.join(test_path, filename)
            if os.path.isfile(file_path):
                write_to_csv(file_path, 'weapons_test.csv')

    # Get min dimensions
    weapons_data = pd.read_csv('weapons_test.csv')
    min_width = weapons_data['Width'].min()
    min_height = weapons_data['Height'].min()
    images_path = 'Sohas_weapon-Detection/images_test/'
    
    img, labs = resize_images(images_path, min_width, min_height)

    resized_images = np.array(img)
    labels = np.array(labs)

    training_X, training_y, testing_X, testing_y = shuffle_split_data(resized_images, labels, 72, 0.80)

    # Input shape of (x, y, color channel)
    input_shape = (min_width, min_height, 3)
    
    neurons = [32, 64, 128, 256]
    k_values = [3, 5, 7, 9]
    rates = [0.0001, 0.001, 0.01, 0.1]
    # hyperparameter_search(neurons, k_values, rates, input_shape, training_X, training_y, testing_X, testing_y, 72)
    train_vggnet(input_shape, training_X, training_y, testing_X, testing_y, 0.0001, 72)

if __name__ == "__main__":
    main()