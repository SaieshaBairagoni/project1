#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data() 


# In[3]:


x_train.shape


# In[7]:


import matplotlib.pyplot as plt
fig,axs=plt.subplots(4,4,figsize=(30,30))
plt.gray()
for i,a in enumerate(axs.flat):
    a.matshow(x_train[i])
    a.axis('off')
    a.set_title('Number {}'.format(y_train[i]))
fig.show()    


# In[21]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
#input_shape=(28,28,1)


# In[22]:


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train/=255
x_test/=255
print('x_train shape:',x_train.shape)
print('number of images in x_train',x_train.shape[0])
print('number of images in x_test',x_test.shape[0])


# In[ ]:





# In[18]:


from tensorflow.python.ops.gen_math_ops import dense_bincount
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D


# In[27]:


model=Sequential()
model.add(Conv2D(28, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


# In[33]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=1)
model.evaluate(x_test,y_test)


# In[ ]:




