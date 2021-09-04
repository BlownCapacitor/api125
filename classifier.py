from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
x_train, x_test, y_train, y_test = train_test_split(X,y, train_size = 7500, test_size = 2500, random_state = 0)
x_trainScale = x_train/255.0
x_testScale = x_test/255.0
l_r = LogisticRegression(solver = 'saga', multi_class= 'multinomial').fit(x_trainScale, y_train)



def getPrediction2(image):
    im_PIL = Image.open(image)
    img_bw = im_PIL.convert('L')
    img_bw_rs = img_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_rs,pixel_filter)
    img_bw_rs_invertedScaled = np.clip(img_bw_rs-min_pixel,0,255)
    max_pixel = np.max(img_bw_rs)
    img_bw_rs_invertedScaled = np.asarray(img_bw_rs_invertedScaled)/max_pixel
    test_sample = np.array(img_bw_rs_invertedScaled).reshape(1,784)
    test_predict = lrPrediction = l_r.predict(x_testScale)
    return test_predict[0]

