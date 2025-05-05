""" Libraries Used:
Pillow : For image processing.
SciPy : For scientific computing.
TensorFlow : A deep learning library that includes Keras for building models."""

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""Conv2D: For the convolutional layer. (Feature extraction:  multiple layer available (Feature:i.e. nose,eyes,face))
MaxPooling2D: For the max pooling layer. (reducing the size of the image)
Flatten: For the flatten layer. (convert 2D data (image) into 1D data (array) and reduce the size of the image)
Dense: For the dense layer. (assign the weightaige to the feutures [i.e. Nose:(True/False=0/1), Eyes:(True/False=0/1), Face:(True/False=0/1)])
Sequential: To define the model sequentially.
Adam Optimizer: To improve model performance. (better efiiciency and faster convergence for binary data)
ImageDataGenerator: To hstore image data  within the Python environment."""


# Define the model
mymodel = Sequential() # Create a sequential model

# Adding different different layers to the model to create a proper convolutional neural network
# Layer 1
mymodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) # Add a convolutional layer with 32 filters of size (3, 3) and ReLU activation function | input_shape is the shape of the input image
mymodel.add(MaxPooling2D()) # Add a max pooling layer to reduce the size of the image
"""A Convolution Layer in a CNN applies filters (kernels) to an image, enabling feature extraction by detecting edges, textures, and patterns essential for image recognition. 🚀
Number of Convolutions (n) follows the power of two pattern:
1st Convolution Layer → 2^1 = 2 filters | 2nd Convolution Layer → 2^2 = 4 filters |3rd Convolution Layer → 2^3 = 8 filters
n-th Convolution Layer → 2^n filters
# (3,3) 3 by 3 is the size of the each convolutional/filter .
# activation is mathematical function applied to the output of the convolutional layer. 
An activation function adds non-linearity to a neural network, helping it to learn complex patterns rather only simple linear mappings. 
Without activation functions, it behave like simple linear models, and difficult to recognize intricate patterns.
# ReLU (Rectified Linear Unit) is a widely used activation function in CNNs. It replaces negative values with zero while keeping positive values unchanged:
Prevents vanishing gradients (helps deep networks learn effectively). Computationally efficient (simple operations: thresholding negatives to zero). 
Improves convergence speed (compared to older functions like sigmoid).
# Input Shape defines the image dimensions expected by the CNN | (64, 64) → Image has a height and width of 64 pixels each.
| 3 → Image contains 3 color channels (RGB), meaning it's a colored image.
"""
# Layer 2
mymodel.add(Conv2D(32, (3, 3), activation='relu')) # No need to specify input shape again as it is already defined in the first layer
mymodel.add(MaxPooling2D()) # Add a max pooling layer to reduce the size of the image

# Layer 3
mymodel.add(Conv2D(32, (3, 3), activation='relu')) # No need to specify input shape again as it is already defined in the first layer
mymodel.add(MaxPooling2D()) # Add a max pooling layer to reduce the size of the image
mymodel.add(Flatten()) # Flatten the image to convert it into a 1D array
mymodel.add(Dense(100, activation='relu')) # Add a dense layer with 100 neurons and ReLU activation function
mymodel.add(Dense(1, activation='sigmoid')) # Add a dense layer with 1 neuron and sigmoid activation function | output layer | sigmoid is used for binary classification (0 or 1) | 1 neuron is used for binary classification (0 or 1)
"""# ✅ Adds a fully connected (Dense) layer with 100 neurons  
# ⮞ Why 100? → Provides sufficient capacity for learning complex patterns while maintaining efficiency.  
# ⮞ What does it do? → Each neuron processes features extracted from previous layers, improving classification accuracy.  
# ⮞ Uses ReLU activation → Prevents vanishing gradients and enhances learning of complex relationships.  

# 🎯 Effect of Reducing or Increasing Neuron Count (100 → Lower or Higher)
# 🔹 Reducing (e.g., 50 neurons):
#    ⮞ Faster processing and lower memory usage ✅
#    ⮞ May struggle to learn intricate features ❌
#    ⮞ Suitable for simpler models with fewer features ✅

# 🔹 Increasing (e.g., 200 neurons):
#    ⮞ Improves feature learning and classification quality ✅
#    ⮞ Requires more computation power, slowing processing ❌
#    ⮞ Risk of overfitting if not enough training data ❌
"""

# Modell Compile
mymodel.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer and binary crossentropy loss function

# Define the Data
train=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True) # Create an image data generator for training data | rescale is used to normalize the image data | shear_range is used to apply shear transformation to the image | zoom_range is used to apply zoom transformation to the image | horizontal_flip is used to flip the image horizontally
test=ImageDataGenerator(rescale=1./255) # Create an image data generator for testing data | rescale is used to normalize the image data
# Load the data
train_img=train.flow_from_directory('dataset_new/Drowsy/train',target_size=(150,150),batch_size=32,class_mode='binary') # Load the training data from the directory | target_size is the size of the image | batch_size is the number of images to be loaded at a time | class_mode is the type of classification (binary in this case)
test_img=test.flow_from_directory('dataset_new/Drowsy/test',target_size=(150,150),batch_size=32,class_mode='binary') # Load the testing data from the directory | target_size is the size of the image | batch_size is the number of images to be loaded at a time | class_mode is the type of classification (binary in this case)

#Train and Test the model
eyes_model=mymodel.fit(train_img,epochs=10,validation_data=test_img) # Train the model with the training data | epochs is the number of times the model will be trained on the data | validation_data is the testing data

# Save the model
mymodel.save('drowsy.h5') # Save the model to a file | 'eyes.h5' is the name of the file where the model will be saved