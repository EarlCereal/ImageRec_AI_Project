import tensorflow as tf
from keras.src.optimizers import Adamax
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt


#Define Path & Categories
train_path = 'Dataset/Training'
val_path = 'Dataset/Validation'

category_names = os.listdir(train_path)
nb_categories = len(category_names)

#Labels
train_images = []

for category in category_names:
    folder = train_path + '/' + category
    train_images.append(len(os.listdir(folder)))


val_images = []
for category in category_names:
    folder = val_path + "/" + category
    val_images.append(len(os.listdir(folder)))

train_men = "Dataset/Training/male"
train_women = "Dataset/Training/female"


#Image Prep
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(300, 300),
        batch_size=20,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
        val_path,
        target_size=(300, 300),
        class_mode='binary')


#Define Model
cnn_model = tf.keras.models.Sequential([

    layers.Conv2D(16, 3, activation='relu', input_shape=(300, 300, 3)), #270.000 input vals / img
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])



cnn_model.compile(loss='binary_crossentropy',
              optimizer=Adamax(learning_rate=0.001),
              metrics=['acc'])


history = cnn_model.fit(
      train_generator,
      steps_per_epoch=20,
      epochs=10,
      verbose=1,
      validation_data=validation_generator)

#TODO: Save weights / model


#Visuals
plt.figure(figsize = (12, 8))
plt.plot(history.history['acc'],label='training accuracy')
plt.plot(history.history['val_acc'],label='valuation accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()

#Prediction
img = tf.keras.utils.load_img('test.png', target_size = (300, 300))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = cnn_model.predict(img_array)
print(prediction)
#TODO: Fix prediction output and organize code



