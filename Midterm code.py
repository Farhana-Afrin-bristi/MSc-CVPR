#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


# In[4]:


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# In[5]:


categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[11]:


plt.figure(figsize= (20,20))
for i in range(100):
    plt.subplot(10,10, i+1)
    plt.imshow(train_images[i])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(train_labels[i][0])
plt.show()


# In[23]:


model = keras.Sequential([
    keras.Input(shape=(32,32,3)),
    keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax'),

])
model.summary()


# In[38]:


model.compile(
     optimizer = keras.optimizers.Adam(learning_rate=0.0001),
     loss = keras.losses.sparse_categorical_crossentropy, 
    metrics=['accuracy']
)


# In[35]:


train_images_norm = train_images.astype('float32') /255
test_images_norm = test_images.astype('float32') /255


# In[42]:


h = model.fit(x=train_images_norm, y=train_labels, epochs=10, batch_size=128, validation_split=0.3)


# In[43]:


plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
plt.plot(h.history['accuracy'], '--o')
plt.plot(h.history['val_accuracy'], '--o')
plt.legend(['Train ACC', 'Validation ACC'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim([0, 11])
plt.ylim([0, 1.5])


plt.subplot(1,2,2)
plt.plot(h.history['loss'], '--o')
plt.plot(h.history['val_loss'], '--o')
plt.legend(['Train Loss', 'Validation Loss'])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[ ]:




