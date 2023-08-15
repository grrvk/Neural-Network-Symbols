#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from extra_keras_datasets import emnist


# In[11]:


(X_train, y_train), (X_test, y_test) = emnist.load_data(type='balanced')


# In[23]:


categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
              'f', 'g', 'h', 'n', 'q', 'r', 't']


# In[40]:


len(categories)


# In[12]:


len(X_train)


# In[33]:


X_train.shape


# In[110]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[111]:


X_test_flattened.shape


# In[491]:


from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(784,)),
        Dense(284, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=None)),
        Dense(184, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=None)),
        Dense(84, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=None)),
        Dense(77, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None)),
        Dense(67, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None)),
        Dense(47, activation='softmax', kernel_initializer=tf.keras.initializers.HeNormal(seed=None))
        ### END CODE HERE ### 
    ], name = "my_model" 
)


# In[492]:


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = 'adam', metrics = ['accuracy']
)

history = model.fit(
    X_train_flattened,y_train,
    epochs=70
)


# In[461]:


image_of_five = X_test_flattened[1]

sample_letter = X_test[1]
sample_letter_resize = sample_letter.reshape(28, 28)
plt.imshow(sample_letter_resize, cmap = 'binary')
plt.axis("off")
plt.show()


prediction = model.predict(image_of_five.reshape(1,784))  # prediction

print(f" predicting a E: \n{prediction}")
print(f" predicting a E: \n{prediction[0][np.argmax(prediction)]}")
print(f" Largest Prediction index: {np.argmax(prediction)}")
print(f" predicting a E: \n{categories[np.argmax(prediction)]}")


# In[462]:


y_predict = model.predict(X_test_flattened)


# In[463]:


np.argmax(y_predict[1])
categories[np.argmax(y_predict[1])]


# In[464]:


y_predicted_labels = [np.argmax(i) for i in y_predict]
y_predict_letters = [categories[np.argmax(i)] for i in y_predict]
y_predicted_labels[:20]


# In[465]:


y_test[:20]


# In[466]:


cm = tf.math.confusion_matrix(labels=y_test, predictions = y_predicted_labels)
cm


# In[467]:


import seaborn as sb
plt.figure(figsize=(47, 47))
sb.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')


# In[ ]:




