import tensorflow as tf
import os
from train_model import process_path


TEST_PICS_DIRECTORY = '../data/external/test_v2'
images = os.listdir(TEST_PICS_DIRECTORY)
image_list = [TEST_PICS_DIRECTORY+i for i in images]  
X_test = tf.data.Dataset.list_files(image_list, shuffle=False).map(process_path) 

model = tf.keras.models.load_model("01_unet_model")
model.predict(X_test)
