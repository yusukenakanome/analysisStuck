## RNN(GRU) による日経平均株価（終値）の予測
## 過去50日分の株価より当日の株価を予測

## 過去700～101日分を訓練用データ
## 過去100～51日分を検証用データ
## 過去50～0日分をテスト用データ

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
lookback = 50

## エポック数
epochs = 4000

## 学習データ数
learning_num = -700

## データファイルを読み込む
## データ日付を index に設定
## 最終行はデータではないため、スキップ
df = pd.read_csv(data_file, index_col=0, encoding='cp932', 
                 skipfooter=1, engine='python')

## 終値
closing_price = df[['終値']].values

## 訓練・検証・テスト用データを作成
## 過去50日分の株価より当日の株価とする
def data_split(data, start, end, lookback):
    length = abs(start-end)
    
    X = np.zeros((length, lookback))
    y = np.zeros((length, 1))
    
    for i in range(length):
        j = start - lookback + i
        k = j + lookback
        
        X[i] = data[j:k, 0]
        y[i] = data[k, 0]
        
    return X, y

## 訓練・検証・テスト用データ
(X_train, y_train) = data_split(closing_price, learning_num, -100, lookback)
(X_valid, y_valid) = data_split(closing_price, -100, -50, lookback)
(X_test, y_test) = data_split(closing_price, -50, 0, lookback)

## 標準化:scalerは正規化するための関数
## X のみ次元を変換（2次元 ⇒ 3次元）
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train).reshape(-1, lookback, 1)
X_valid_std = scaler.transform(X_valid).reshape(-1, lookback, 1)
X_test_std = scaler.transform(X_test).reshape(-1, lookback, 1)


scaler.fit(y_train)
y_train_std = scaler.transform(y_train)
y_valid_std = scaler.transform(y_valid)

#X_train_std.shape[-1] 行列X_train_stdの列数
length_of_sequence = X_train_std.shape[-1]
#隠れ層の数
n_hidden = 256

## 訓練 RNN(ここでmodelを作る)
model = Sequential()
model.add(GRU(n_hidden, 
              dropout=0.2, 
              recurrent_dropout=0.2, 
              return_sequences=False, 
              input_shape=(None, X_train_std.shape[-1])))
model.add(Dense(1)) ##ニューロン数

model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])

result = model.fit(X_train_std, y_train_std, 
                   verbose=0,   ## 詳細表示モード
                   epochs=epochs, 
                   batch_size=64, 
                   shuffle=True, 
                   validation_data=(X_valid_std, y_valid_std))

## 訓練の損失値をプロット
epochs = range(len(result.history['loss']))
plt.title('損失値（Loss）')
plt.plot(epochs, result.history['loss'], 'bo', alpha=0.6, marker='.', label='train', linewidth=1)
plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='valid', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

## 予測値
df_predict_std =  pd.DataFrame(model.predict(X_test_std), columns=['予測値'])

## 予測値を元に戻す(正規化の解除)
predict = scaler.inverse_transform(df_predict_std['予測値'].values)

## 予測結果をプロット
pre_date = df.index[-len(y_test):].values
plt.title('実際の終値と予測値')
plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label='実際の終値', linewidth=1)
plt.plot(pre_date, predict, 'r', alpha=0.6, marker='.', label='予測値', linewidth=1)
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()

# RMSEの計算
print('二乗平均平方根誤差（RMSE） : %.3f' %  
       np.sqrt(mean_squared_error(y_test, predict)))

#未来の株価予想
future_test = X_test_std[-1].T
# 1つの学習データの時間の長さ
time_length = future_test.shape[1]
# 未来の予測データを保存していく変数
future_result = np.empty((1))

#10日間の日経平均の予測をする
for step2 in range(10):
    #future_testを3次元に変換
    test_data = np.reshape(future_test, (1, time_length, 1))
    #予測値をbatch_predict_stdに格納
    batch_predict_std = model.predict(test_data)
    #batch_predict_stdの正規化を解除する（step2日目の予測）
    batch_predict = scaler.inverse_transform(batch_predict_std)
    #最初の要素を消す
    future_test = np.delete(future_test, 0)
    #future_testの最後尾にbatch_predict_stdを追加する
    future_test = np.append(future_test, batch_predict_std)
    #future_resultにstep2日目のデータを追加
    future_result = np.append(future_result, batch_predict)

# 未来の予測データを保存していく変数作成時に出来た不要な最初の要素を削除
future_result = np.delete(future_result, 0)

## 予測結果をプロット
pre_date = df.index[-len(y_test + future_result):].values
plt.title('未来の予測値', fontname="MS Gothic")
plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label='Real', linewidth=1)
plt.plot(pre_date, predict, 'r', alpha=0.6, marker='.', label='predict', linewidth=1)
plt.plot(range(len(predict) , len(future_result) + len(predict)), future_result, 'g', alpha=0.6, marker='.', label='future_predict', linewidth=1)
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()