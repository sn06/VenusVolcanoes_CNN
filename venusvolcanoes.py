# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 11:51:39 2018

@author: sn06
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator

adm = Adam(lr=0.0001)

train = pd.read_csv('train_images.csv')
train_lab = pd.read_csv('train_labels.csv')
train_lab = train_lab.iloc[1:len(train_lab),:]
test = pd.read_csv('test_images.csv')
test_lab = pd.read_csv('test_labels.csv')
test_lab = train_lab.iloc[1:len(test_lab),:]

X_train = train.values
X_test = test.values
y_train = train_lab['Volcano?'].values.reshape(-1,1)
y_test = test_lab['Volcano?'].values.reshape(-1,1)

del(train,train_lab,test,test_lab)

def find_missing(arr):
    temp=[]
    for i in range(len(arr)):
        for j in range(110):
            if (np.zeros(shape=(110,1)).astype(int)==arr[i][j]).all() == True:
               temp.append(i)     
               break
    return temp

train_missing = find_missing(X_train)
X_train = np.delete(X_train,(train_missing),axis=0)
y_train = np.delete(y_train,(train_missing),axis=0)

test_missing = find_missing(X_test)
X_test = np.delete(X_test,(test_missing),axis=0)
y_test = np.delete(y_test,(test_missing),axis=0)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(len(X_train),110,110,1)
X_test = X_test.reshape(len(X_test),110,110,1)

def plot_image(n):
    plt.imshow(X_train[n].reshape(110,110),cmap='plasma')

def create_model():
    model = Sequential()
    model.add(Conv2D(110,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(110,110,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(220,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(220,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(110,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=adm,loss='binary_crossentropy',metrics=['accuracy'])
    return model

if 'model' not in globals():
    model = create_model()
    
dg = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
dg.fit(X_train)

history = model.fit_generator(dg.flow(X_train,y_train,batch_size=16),steps_per_epoch=int(np.ceil(len(X_train)/32)),epochs=5,validation_data=(X_test,y_test))
model.evaluate(X_test,y_test)
plt.plot(history.history['acc'],label='acc')
plt.plot(history.history['val_acc'],label='v_acc')