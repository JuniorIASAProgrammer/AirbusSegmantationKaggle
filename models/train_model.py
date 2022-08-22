import tensorflow as tf
import keras.backend as K
import os
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def mask_converter(values):
    mask = np.zeros((768*768,), dtype=float)        #create empty one-dimentional vector with zeros
    if isinstance(values, str):
        values = values.strip().split()
        start_points = values[0::2]               #separate values
        lengths = values[1::2]
        for st_p, l in zip(start_points, lengths):     #fill mask with ones according to the EncodedPixels colomn
            st_p, l = int(st_p)-1, int(l)
            ones = np.ones(l, dtype=int) 
            mask[int(st_p):int(st_p)+int(l)] = ones
    return mask.reshape((768, 768,1))


# CNN
# encoder
def conv_block(inputs=None, n_filters=32, max_pooling=True):
    conv = Conv2D(n_filters, kernel_size=3, 
                activation='relu', 
                padding='same', 
                kernel_initializer='he_normal')(inputs)

    conv = Conv2D(n_filters, kernel_size=3, 
                activation='relu', 
                padding='same', 
                kernel_initializer='he_normal')(conv)
    if max_pooling:
        next_layer = MaxPooling2D(2)(conv)
    else:
        next_layer = conv
    skip_connection = conv          #save skip-connection for further usage in decoding
    return next_layer, skip_connection


# decoder
def upsampling_block(previous_layer, prevoius_skip_layer, n_filters=32):
    upsampling = Conv2DTranspose(n_filters,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same')(previous_layer)
    merge = concatenate([upsampling, prevoius_skip_layer])
    conv = Conv2D(n_filters,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)
    return conv


# initializing U-net model
def unet_model(input_size=(768,768,3), n_filters=32, n_classes=2):
    inputs = Input(input_size)
    # downsampling
    c1, skip1 = conv_block(inputs, n_filters)
    c2, skip2 = conv_block(c1, n_filters*2)
    c3, skip3 = conv_block(c2, n_filters*4)
    c4, skip4 = conv_block(c3, n_filters*8)
    c5, _ = conv_block(c4, n_filters*16, max_pooling=False) 

    # uplampling
    c6 = upsampling_block(c5, skip4, n_filters*8)
    c7 = upsampling_block(c6, skip3, n_filters*4)
    c8 = upsampling_block(c7, skip2, n_filters*2)
    c9 = upsampling_block(c8, skip1, n_filters)

    # output
    c10 = Conv2D(n_classes, kernel_size=1, padding='same')(c9)
    return Model(inputs=inputs, outputs=c10)


# initializing dice metric
def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)


# initializing dice metric
def dice(target, pred):
  smooth = 1.0
  intersection = K.sum(target * pred, axis=[1,2,3])
  union = K.sum(target, axis=[1,2,3]) + K.sum(pred, axis=[1,2,3])
  return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def process_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [768,768,1])

    

# Constants initialization
CSV_SHIP_PATH = 'data/processed/train_ship_segmentations_grouped.csv'
TRAIN_PICS_DIRECTORY = 'data/external/train_v2/'
EPOCHS = 5
BATCH_SIZE = 4

# Create U-net
unet = unet_model()
unet.compile(optimizer='adam',
              loss=dice_loss,
              metrics=[dice])
unet.summary()

## ETL
labels_file = pd.read_csv(CSV_SHIP_PATH)            # Read preprocessed csv file
images, labels = labels_file['ImageId'], labels_file['EncodedPixels'].to_numpy()            # Extract labels and images with ships
image_list = [TRAIN_PICS_DIRECTORY+i for i in images]           # Create list of images
X = tf.data.Dataset.list_files(image_list, shuffle=False)           # Convert data into tf.Dataset
y = tf.data.Dataset.from_tensor_slices(labels)
X = X.map(process_path)             
y = y.map(mask_converter)           # Convert start points and run length into mask
train_dataset = tf.data.Dataset.zip((X, y))             # Merge images and masks together
train_dataset = train_dataset.batch(BATCH_SIZE)             # Set batch size
model_history = unet.fit(train_dataset, epochs=EPOCHS)          # Fit model
unet.save('01_unet_model')             # Save model to use             
