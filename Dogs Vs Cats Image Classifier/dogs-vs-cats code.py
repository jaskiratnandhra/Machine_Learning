import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import RMSprop

model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1/255)
test_datagen=ImageDataGenerator(rescale=1/255)
train_dir='C:/Users/User/Desktop/python/ML/NN/train'
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
test_dir='C:/Users/User/Desktop/python/ML/NN/test1'
validation_generator=train_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=15,validation_data=validation_generator,validation_steps=50,verbose=2)

#prediction
path='C:/Users/User/Desktop/python/ML/NN/test1/test/12244.jpg'
img=Image.open(path)
img=img.resize((150,150))
x=np.array(img)
x=np.expand_dims(x, axis=0)
images=np.vstack([x])
classes=model.predict(images,batch_size=10)
print(classes[0])
if(classes[0]>0):
    print("Its a dog")
else:
    print("Its a cat")