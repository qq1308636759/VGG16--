from tensorflow.keras.models import *
import pandas as pd
import cv2
import numpy as np
def model():
    model = load_model('model1.h5')
    return model

def read(path):

    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = img / 255
    img = img.reshape(1, 256, 256, 3)
    return img
def pre(model,img):
    pred = model.predict(img)
    y = np.argmax(pred, axis=-1)
    labels= {0:'daisy',1:'dandelion',2:'rose',3:'sunflower',4:'tulip'}
    y = pd.DataFrame(y)
    y[0]=y[0].map(labels)
    y = y.values.flatten()
    print('此花为：',y)
    return y

if __name__ == '__main__':
    path = r'flowers/24d93_5b65.jpg'
    img = read(path)
    model = model()
    pred = pre(model,img)
    print(pred)
