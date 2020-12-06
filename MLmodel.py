import numpy as np  # For numerical computation
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For data manipulation
import os # For file manipulation
import keras # For creating CNNs

from sklearn.model_selection import train_test_split # To slpit training data into train and validation set
from keras.utils import to_categorical # For converting labels into their one-hot representations

from keras.models import Sequential # Sequential model is a stack of layers
from keras.layers import Conv2D, MaxPooling2D # Convolutional and Maxpooling layers for CNNs
from keras.layers import Dense, Dropout # Dense-Densly connected NN layer, Dropout-Reduces overfittiing
from keras.layers import Flatten, BatchNormalization # Adds a channel dimension to the input


# Importing the training and test dataset
train_df = pd.read_csv()
test_df = pd.read_csv()

train_df.head()

# converting all the columns other than label into a numpy array
train_data = np.array(train_df.iloc[:, 1:])
test_data = np.array(test_df.iloc[:, 1:])

# Converting all the labels into categorical labels
train_labels = to_categorical(train_df.iloc[:, 0])
test_labels = to_categorical(test_df.iloc[:, 0])

rows, cols = 28, 28 # Size of images

# Reshaping the test and train data
train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)
test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)

# To cast data into float32 type
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Normalizing the images
train_data /= 255.0
test_data /= 255.0

# Splitting data into training and validation set
train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.2)

batch_size = 256
epochs = 5
input_shape = (rows, cols, 1)

# Creating the model
def baseline_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

model = baseline_model()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y))

hist = history.history

fig, ax = plt.subplots(2)

ax[0].plot(hist['acc'])
ax[0].plot(hist['val_acc'])
ax[0].legend(['training accuracy', 'validation accuracy'])

ax[1].plot(hist['loss'])
ax[1].plot(hist['val_loss'])
ax[1].legend(['training loss', 'validation loss'])

for axs in ax.flat:
    axs.label_outer()

predict = model.predict(test_data)
