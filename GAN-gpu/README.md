### contributor name
Hirofumi Beppu
### project start date 
2021.10.22
### model name
Generative Adversarial Network
### descriptions
Fake image generator by GAN

# 敵対的生成ネットワークによる実在しない画像データの生成

[GAN][gan]は生成モデルの一種であり、データから特徴を学習することで、実在しないデータを生成したり、存在するデータの特徴に沿って変換できる。

[gan]:https://www.imagazine.co.jp/gan%EF%BC%9A%E6%95%B5%E5%AF%BE%E7%9A%84%E7%94%9F%E6%88%90%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%A8%E3%81%AF%E4%BD%95%E3%81%8B%E3%80%80%EF%BD%9E%E3%80%8C%E6%95%99%E5%B8%AB/

主に参考にしたサイト：<https://www.inoue-kobo.com/ai_ml/gan-pikachu/index.html>

オフィシャルサイト：<https://www.tensorflow.org/tutorials/generative/dcgan?hl=ja>

## 1. 準備（ソフトウェア等）
### 今回使ったプログラムを実行したPCのスペック

・プロセッサ：Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz   3.60 GHz

・実装RAM：32.0 GB

・システムの種類：64 ビット オペレーティング システム、x64 ベース プロセッサ

・OS：Windows 10 Pro

・GPU：NVIDIA GeForce RTX 2070

### 環境

・PyThon 3.7.9（3.8以上だとTensorFlow 2.x以降しか使えないが，今回のプログラムはTensorFlow 1.xを使う必要がある）

・TensorFlow-gpu 1.14.0（tensorflow.contribはTensorFlow 2.x以降では互換性がないため．2.x以降でこのようになったツールが多いが，オープンソースの多くがTensorFlow 1.xをベースに作っているものも未だに多く，また安定して使えるものの多くが1.xにある．）


```
pip install tensorflow-gpu==1.14
```

・Keras 2.1.2（keras_adversarialが使えるのはこれくらいのバージョンらしく，2.1.3以降だとAttributeError: 'Model' object has no attribute 'internal_input_shapes'が出る．）

```
pip install keras==2.1.2
```

### GPUを使うためのツール

・[CUDA 10.0][cuda]（CUDA 11.xに対応しているのはTensorFlow 2.4以降だが，上記の理由により今回は使えない．）

[cuda]:https://developer.nvidia.com/cuda-10.0-download-archive

ちなみに研究室のPCのGPUはGeForce RTX 30シリーズだが，CUDA 11.x以降でないとGPUがうまく認識されない現象があるため要注意．


（参照）

<https://teratail.com/questions/331673>

<https://teratail.com/questions/312270>

<https://teratail.com/questions/323574>

<https://github.com/DeepLabCut/DeepLabCut/issues/944>


### 画像認識のためのツール

・[cuDNN v7.6.5 for CUDA 10.0][cudnn]（CUDA 10.0に対応している最新バージョン．cuDNNはNVIDIAが提供しているPython用の画像認識ツール．）

[cudnn]:https://developer.nvidia.com/rdp/cudnn-archive

### GANを実行するためのプログラム

・keras_adversarial

・utils

ダウンロード先：<https://github.com/bstriner/keras-adversarial>

### 注意

・バージョン管理ができていないと頻繁にエラーが出て苦しむので要注意．

・必要に応じてダウングレードする（python 3.7.9のコマンドプロンプト上でのpip install~だとそのpythonに紐づいたものだけが変わるので，専用のものだと割り切れば楽か．）

・逐次必要なツールはネットからダウンロード or pipでインストールしてください．

## 2. Googleから画像の収集
1. Google画像検索から画像を集めるためのツールを以下でインストール．

```
pip install google_images_download
```

2. ```google_images_download```だけだとダウンロード枚数に制限があるので，[ChromeDriver][chromedriver]もインストール．

[chromedriver]:https://chromedriver.chromium.org/downloads

インストール方法（パスの通し方など）の詳細：<https://www.mittsu-kosen.com/chromedriver%E3%82%92windows10%E3%81%A7%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95%E3%80%90%E7%94%BB%E5%83%8F%E4%BB%98%E3%81%8D%E3%80%91/>

3. [downloader.ipynb](./downloads/downloader.ipynb)を実行．今回はピカチュウの画像をダウンロード．

## 3. 画像サイズの統一化
1. [reshape.ipynb](./reshape_labels/reshape.ipynb)を実行．


## 4. 欲しい画像の抽出
ダウンロードしてきた画像には本来学習させたい画像（今回はピカチュウ）以外のものも含まれるので，ここではネットワークにいくつか画像を学習させ，自動的に欲しい画像を抽出させてくる方法を記載する．

1. [labeled](./dataset/labeled)に訓練用と検証用データを置く．それぞれに欲しいもの（ピカチュウ）とそうでないものに分けていくつか教師データを用意する．（このあたりは手作業．）また保存先のフォルダを事前に用意（今回だとpredicted_auto2）．

2. [train.ipynb](./find/train.ipynb)を実行．

## 5. GANの学習

1. [generate.ipynb](./generate.ipynb)を実行．[generated](./generated)のフォルダに100エポックごとの（偽の）生成画像が保存されていく．

### 注意

この環境だと1エポックあたりおよそ1sだったが，cpuで計算したところ1エポックあたり20sほどで50000エポック計算しようと思うと，10日以上かかることになるので，やはりgpuを使わないと厳しいかもしれない．
