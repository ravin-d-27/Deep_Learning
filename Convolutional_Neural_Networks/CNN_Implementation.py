
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Note: Feature Scaling is Compulsary for implementing Neural Networks'

'_________________________________________________________________________________________________________________________________________'

"Data Preprocessing for the training set"

train_datagenerator = ImageDataGenerator(

    rescale=1./255, # This is the feature scaling for each and every pixel of the image dataset
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True

)

data_train = train_datagenerator.flow_from_directory(

    'D:/My_Github_New/Deep_Learning/Convolutional_Neural_Networks/dataset/training_set',
    target_size=(100,100), # Final size of the images for feeding it into Convolutional Neural Networks 
    batch_size=32, # This is the default value of batch and its okay to run with it
    class_mode='binary'

) # Binary Output will be given by this model. Which is either CAT or DOG

'_________________________________________________________________________________________________________________________________________'

"Data Preprocessing of the Test Dataset"

test_datagenerator = ImageDataGenerator(rescale=1./255)

data_test = train_datagenerator.flow_from_directory(

    'D:/My_Github_New/Deep_Learning/Convolutional_Neural_Networks/dataset/test_set',
    target_size=(100,100), 
    batch_size=32, 
    class_mode='binary'

)

'_________________________________________________________________________________________________________________________________________'

"Building the Architechure of Convolutional Neural Networks"

cnn = tf.keras.models.Sequential()

# Convolution

cnn.add(tf.keras.layers.Conv2D( # We are adding a convolutional Layer Now
    
    filters = 32, # You are free to use as many filters as you can
    kernel_size = 3, #  It is the size of the filters. 
    activation = "Relu", # Relu means Rectified Linear Unit and it is used for classification purpose
    input_shape = [100,100,3] # We are adding 3 at index 2 for feeding colour images

)) 

# Pooling

cnn.add(tf.keras.models.MaxPool2D(
    
    pool_size=2,
    strides = 2,
    
))

# Adding the second Convolutional Layer - Use the same code for making this second convolutional layer

# The input_shape parameter can be removed because it is only applicable for the first convolutional layer

cnn.add(tf.keras.layers.Conv2D( 
    
    filters = 32, 
    kernel_size = 3, 
    activation = "Relu",

)) 

cnn.add(tf.keras.models.MaxPool2D(
    
    pool_size=2,
    strides = 2,
    
))

# Flattening 

cnn.add(tf.keras.layers.Flatten())

# Fully Connected Neural Network

cnn.add(tf.keras.layers.Dense(

    units = 128,
    activation = "relu"

))


cnn.add(tf.keras.layers.Dense( # Output Layer

    units = 1, # Because it gives classification for either one object
    activation = "sigmoid"

))

'_________________________________________________________________________________________________________________________________________'




