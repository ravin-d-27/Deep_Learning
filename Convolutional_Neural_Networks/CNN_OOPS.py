
from importlib.resources import path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image
        
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CNN_DL:

    def __init__():
        pass

    def train_and_test_data_preprocess(self,train_path, test_path):

        train_datagenerator = ImageDataGenerator(

        rescale=1./255,
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True

        )

        self.data_train = train_datagenerator.flow_from_directory(

        train_path,
        target_size=(100,100),
        batch_size=32,
        class_mode='binary'

        )

        test_datagenerator = ImageDataGenerator(rescale=1./255)

        self.data_test = train_datagenerator.flow_from_directory(

        test_path,
        target_size=(100,100), 
        batch_size=32, 
        class_mode='binary'

        )

    def CNN_Architecture(self):

        self.cnn = tf.keras.models.Sequential()

        self.cnn.add(tf.keras.layers.Conv2D(
            
            filters = 32,
            kernel_size = 3,
            activation = "relu",
            input_shape = [100,100,3]

        )) 

        self.cnn.add(tf.keras.layers.MaxPool2D(
            
            pool_size=2,
            strides = 2,
            
        ))

        self.cnn.add(tf.keras.layers.Conv2D( 
            
            filters = 32, 
            kernel_size = 3, 
            activation = "relu",

        )) 

        self.cnn.add(tf.keras.layers.MaxPool2D(
            
            pool_size=2,
            strides = 2,
            
        ))

        self.cnn.add(tf.keras.layers.Flatten())

        self.cnn.add(tf.keras.layers.Dense(

            units = 128,
            activation = "relu"

        ))


        self.cnn.add(tf.keras.layers.Dense( 

            units = 1, 
            activation = "sigmoid"

        ))


    def train(self):
        self.cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.cnn.fit(x = self.data_train, validation_data = self.data_test, epochs = 25)


    def predict(self,predict_path):
        test_image = image.load_img(predict_path, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = self.cnn.predict(test_image)
        self.data_train.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        print(prediction)



