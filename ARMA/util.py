# https://qiita.com/DS27/items/1e998a58488e76bfcbdc
# 必要なライブラリーのインポート
import pandas as pd
import numpy
import math
import os

def devide_data(file_name, file_type, num_train_data = False):   
    """
    devide_data
    ・ファイルデータを読み込み、学習用データとテスト用データに分割
    ・ファイル拡張子はcvsのみ対応
    ・ファイルは同じ階層に置く
    
    引数：
        file_name (string)   : 読み込むファイル名
        file_type (string)   : 拡張子
        num_train_data (int) : 学習データ数、指定しない場合、8:2で分割
    """
    
    sep = os.sep;# separator('\' for WINDOWS)
    extsep = os.extsep;# file extension separator
    
    if file_type == 'csv':
        
        # データの読み込み
        file_path = file_name + extsep + file_type
        df = pd.read_csv(file_path)

        num_data = df.shape[0]
        
        # dataframeを学習用とテスト用に分割
        if not num_train_data:
            num_train_data = math.floor(0.8*num_data)

        df_train = df.head(num_train_data)
        df_test = df.tail(num_data - num_train_data)

        # データの中身を確認
#         print(df.shape)
#         print(df_train.shape)
#         print(df_test.shape)

#         print(df_train.tail())
#         print(df_test.head())

        # dataframeをcvsに書き込む
        df_train.to_csv('train'+ sep + file_path, index = False)
        df_test.to_csv('test'+ sep + file_path, index = False)
        
    else:
        print('the file format {} is not supported'.format(file_type))