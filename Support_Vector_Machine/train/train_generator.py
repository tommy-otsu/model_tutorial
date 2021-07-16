
#import
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#データのロードと訓練-テストデータへの分割
data_set = load_iris() 
df = pd.DataFrame(data_set.data, columns=data_set.feature_names)
y  = data_set.target
x = df 
#データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# モデルに訓練用データを与えて学習させる
clf = SVC()
clf.fit(x_train, y_train)
 
# テストデータを評価する
y_pred = clf.predict(x_test)
ss = accuracy_score(y_test, y_pred)
print(ss)
