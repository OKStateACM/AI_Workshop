import tflearn
import tflearn.datasets.mnist as mnist #name it mnist so we don't need to spell out tflearn.datasets.mnist all the time

# build the network
input_layer = tflearn.input_data(shape=[None, 784])
hidden_layer = tflearn.fully_connected(input_layer, 100, activation='tanh')
output_layer = tflearn.fully_connected(hidden_layer, 10, activation='softmax')

X, Y, testX, testY = mnist.load_data(one_hot=True) #loads the training data, X,Y and the test data testX,testY

# Regression using SGD with learning rate decay
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000) #create the optimizer function
accuracy = tflearn.metrics.Accuracy() #create a function to define our model's accuracy. (use tflearn's builtin)
trainer = tflearn.regression(output_layer, optimizer=sgd, loss='mean_square', metric=accuracy)

# Training
model = tflearn.DNN(output_layer, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY), show_metric=True)
