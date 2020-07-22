import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os 
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


local_zip = '/content/drive/My Drive/ai/st.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/st')
local_zip = '/content/drive/My Drive/ai/st-validation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/st-validation')
zip_ref.close()

train_shreyash_dir = os.path.join('/st/shreyash')

train_tejas_dir = os.path.join('/st/tejas')

validation_shreyash_dir = os.path.join('/st-validation/shreyash')

validation_tejas_dir = os.path.join('/st-validation/tejas')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])
              
 
train_datagen = ImageDataGenerator(rescale=1/255)

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        '/st/', 
        target_size=(300, 300),
        batch_size=30,
        class_mode='binary'
        )

validation_generator = validation_datagen.flow_from_directory(
        '/st-validation/',
        target_size=(300, 300),
        batch_size=30,
        class_mode='binary')

history = model.fit(
      train_generator,
      #steps_per_epoch=3,  
      epochs=15,
      verbose=1,
      #validation_data = validation_generator,
      #validation_steps=8
      )
      
model.save('detect_model.h5')
