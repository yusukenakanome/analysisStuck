import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

## 以下の URL より日経平均株価データ（日別）をダウンロードし、
## このプログラムと同一階層に保存
## https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_jp.csv

data_file = 'nikkei_stock_average_daily_jp.csv'

## 当日の株価を予測するために必要な過去の日数
lookback = 30

## エポック数
epochs = 5000

## 学習データ数
learning_num = -750

## データファイルを読み込む
## データ日付を index に設定
## 最終行はデータではないため、スキップ
df = pd.read_csv(data_file, index_col=0, encoding='cp932', 
                 skipfooter=1, engine='python')

## 終値
closing_price = df['終値'].values

## 欠損値の確認
print('欠損値の個数')
print(df.isnull().sum(), '\n')

## 基本統計量の確認（終値）
print('終値の基本統計量')
print(df['終値'].describe(), '\n')
"""
## 終値を時系列にプロット
plt.title('日経平均株価（終値）の推移', fontname="MS Gothic")
plt.plot(range(len(closing_price)), closing_price)
plt.show()

plt.title('日経平均株価（終値）の推移　過去300日分のみ', fontname="MS Gothic")
plt.plot(range(len(closing_price[-300:])), closing_price[-300:])
plt.show()

plt.title('日経平均株価（終値）の推移　過去30日分のみ', fontname="MS Gothic")
plt.plot(df.index[-30:], closing_price[-30:])
plt.xticks(rotation=70)
plt.show()
"""
## 訓練・検証・テスト用データを作成
## 過去30日分の株価より当日の株価とする←ここが怪しい   
def data_split(data, start, end, lookback):
    length = abs(start-end)
           
    X = np.zeros((length, lookback))
    y = np.zeros((length, 1))
    for i in range(length):
        j = start - lookback + i
        k = j + lookback
        
        X[i] = data[j:k] ##学習データ
        y[i] = data[k] ##正データ

    return X, y

##次の日の株価予想としてZ[3日前~本日]を用意、それをモデルにぶち込むので、ますはZ[]を作る
def data_split2(data, start, end, lookback):
    length = abs(start-end)
          
    X = np.zeros((length, lookback))
    y = np.zeros((length, 1))
    for i in range(length):
        j = start - lookback + i + 1
        k = j + lookback
        
        X[i] = data[j:k] if k < 0 else data[j:] ##学習データ

    return X, y


## 訓練・検証・テスト用データ・未来データ
(X_train, y_train) = data_split(closing_price, learning_num, lookback * -2, lookback)
(X_valid, y_valid) = data_split(closing_price, lookback * -2, lookback * -1, lookback)
(X_test, y_test) = data_split(closing_price, lookback * -1, 0, lookback) ##ここのデータセットにラスト4つの値をいれたい
(X_today_predict, y_sample) = data_split2(closing_price, lookback * -1, 0, lookback)

## 訓練
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[-1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])

result = model.fit(X_train, y_train, 
                   verbose=0,   ## 詳細表示モード
                   epochs=epochs, 
                   batch_size=64, 
                   shuffle=True, 
                   validation_data=(X_valid, y_valid))

## 訓練の損失値をプロット
epochs = range(len(result.history['loss']))
plt.title('損失値（Loss）', fontname="MS Gothic")
plt.plot(epochs, result.history['loss'], 'bo', alpha=0.6, marker='.', label='Training_loss', linewidth=1)
plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='Verification_loss', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 2000)
plt.show()

## 予測値
df_predict =  pd.DataFrame(model.predict(X_test), columns=['予測値']) ##テストデータをモデルにぶち込む
df_predict2 =  pd.DataFrame(model.predict(X_today_predict), columns=['予測値']) ##明日の日経平均用

## 予測結果をプロット，df_predict['予測値'].values←これが予測値
pre_date = df.index[-len(y_test):].values ##実際のデータ
plt.title('実際の終値と予測値', fontname="MS Gothic") ##表題
plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label='Real', linewidth=1) ##実際のデータプロット
plt.plot(pre_date, df_predict['予測値'].values, 'r', alpha=0.6, marker='.', label='Predicted', linewidth=1) ##予測データプロット
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()

##print('今日の日経平均は:',y_test[lookback - 1])
print('明日の日経平均予測は:', df_predict2.values[len(df_predict2) - 1])

## RMSEの計算
print('二乗平均平方根誤差（RMSE） : %.3f' %  
       np.sqrt(mean_squared_error(y_test, df_predict['予測値'].values)))




