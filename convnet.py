# Import tflearn and sub components
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data() # load the data from tflearn package
X, Y = shuffle(X, Y) # randomize the image order. like shuffling a deck of cards
Y = to_categorical(Y, 10) # like the one_hot encoding used for MNIST data
Y_test = to_categorical(Y_test, 10) # like the one_hot encoding used for MNIST data

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center() # 
img_prep.add_featurewise_stdnorm()     # 

# Convolutional network building
input = input_data(shape=[None, 32, 32, 3], #our image data is 32x32 pixels with 3 color channels, RGB
                 data_preprocessing=img_prep,
                 data_augmentation=img_aug)
conv_layer_1 = conv_2d(input, 32, 3, activation='relu') #apply a 3x3 filter on our image, and we will use 32 different filters at this layer (that means the output is a 32x32x32 vector, one 'channel' for each filter)
maxpool_1 = max_pool_2d(conv_layer_1, 2) #maxpool with a 2x2 pooling region size (now 16x16x32)
conv_layer_2 = conv_2d(maxpool_1, 64, 3, activation='relu') # apply a 3x3 filter on our image, with 64 filters (now 16x16x64)
conv_layer_3 = conv_2d(conv_layer_2, 64, 3, activation='relu') # # apply a 3z3 filter on our image, with 64 filters (still 16x16x64)
maxpool_2 = max_pool_2d(conv_layer_3, 2) # maxpool with a 2x2 pooling region size (now 8x8x64)
fully_connected_1 = fully_connected(maxpool_2, 512, activation='relu') # layer of 512 rectified linear units
fully_connected_1 = dropout(fully_connected_1, 0.5) # dropout rate of 50%
output = fully_connected(fully_connected_1, 10, activation='softmax') # final layer, one node for each of the 10 image classes. we use softmax as with the MNIST classifier
optimizer = regression(output, optimizer='adam', #Adam is an improvement on SGD, however the principal is the same.
        loss='categorical_crossentropy', # rather than mean-squared error, we use a different loss function 
        learning_rate=0.001)


# Train using classifier. shuffle the data, and use train with a batch size of 96 images at a time.
model = tflearn.DNN(optimizer, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=96, run_id='cifar10_cnn')
