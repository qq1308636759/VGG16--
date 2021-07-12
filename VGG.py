import pandas as pd
import numpy as np
from keras.models import *
from keras.applications import ResNet50,VGG16,MobileNet,InceptionResNetV2,ResNet50V2,ResNet101,ResNet101V2,ResNet152,ResNet152V2,InceptionV3,NASNetLarge
import os
from keras import layers, optimizers, models
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import keras
import cv2
import glob
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

names = ['daisy','dandelion','rose','sunflower','tulip']
# X = []
# Y = []
# for i in names:
#     # 遍历文件夹，读取图片
#     for f in os.listdir(r"flowers/{}".format(i)):
#         print(f)
#         Images = cv2.imread(r"flowers/{}/{}".format(i, f))  # images[i,j,k]
#         # resize函数图像缩放
#         image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC-基于4x4像素邻域的3次插值法
#         X.append(image)
#         Y.append(i)
# X = np.array(X)
# Y = np.array(Y)
# print(X)
# print(Y)
# np.save('x.npy',X)
# np.save('y.npy',Y)
X_path = 'x.npy'
Y_path = 'y.npy'
X = np.load(X_path)
Y = np.load(Y_path)
labels= {'daisy':0,'dandelion':1,'rose':2,'sunflower':3,'tulip':4}
Y = pd.DataFrame(Y)
Y[0]=Y[0].map(labels)
Y = Y.values.flatten()
Y = to_categorical(Y, 5)
X = X/255
print(X.shape)
print(X)
print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=1)


def model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = models.Sequential()
    model.add(conv_base)
    # model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    conv_base.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001),
                  metrics=['categorical_accuracy'])
    model.summary()
    return model
model=model()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
model_checkpoint = ModelCheckpoint('model2.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=50, batch_size=32,validation_data=(x_test,y_test),callbacks=[early_stop,model_checkpoint])
model.save("model1.h5")

model=load_model('model1.h5')
pred = model.predict(x_test)
y = np.argmax(pred, axis=-1)
y_test = np.argmax(y_test, axis=-1)
print(confusion_matrix(y_test, y))
print(classification_report(y_test, y))
cm = confusion_matrix(y_test, y)
print(cm)
plt.imshow(cm, cmap=plt.cm.BuPu)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(cm))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
#plt.xticks(indices, [0, 1, 2])
#plt.yticks(indices, [0, 1, 2])
label_name = ['daisy','dandelion','rose','sunflower','tulip']
ax = plt.gca()
plt.xticks(indices,label_name,fontsize=8)
ax.xaxis.set_ticks_position("top")
plt.yticks(indices, label_name,fontsize=8)
plt.colorbar()
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#
# 显示数据
for first_index in range(len(cm)):    #第几行
    for second_index in range(len(cm[first_index])):    #第几列
        plt.text(first_index, second_index, cm[first_index][second_index],fontdict={'size':6})
# 显示
plt.show()
# ----------------------------------------------------------------------------------------------