### ・contributor name
Nishida Shuhei
### ・project start date 
2021.07.21
### ・model name
statsmodels.tsa.arima.model
### ・descriptions
Time series prediction by ARMA model

# ARMAモデルによる時系列予測

自己回帰移動平均モデル（autoregressive moving average model, ARMAモデル）は時系列データに適用される線形モデルの一種で、直近数ステップでの観測や誤差から現在の観測を説明する。


ARMA_pred.ipynbでは、以下の手順でARMAモデルを用いた時系列予測を行った。

## 1. 前処理
単位根検定を行い、必要に応じて階差系列を取るなど前処理を行う
## 2. モデル次数の決定
コレログラムやかばん検定の結果を見ながら、ARMAの次数を決める
## 3. フィッティング
最尤推定によりモデルの係数を求める
## 4. 予測
フィッティングしたモデルを使って将来の観測値を予測する

航空機搭乗者数のデータに適用した印象
* データの大雑把な傾向をつかむのはうまいが、高精度もしくは長期の予測は難しい
* 次数を大きくすると、フィッティングがうまくいかないので、ステップ2では、自己相関がありそうでも妥協して小さい次数を選んだ
