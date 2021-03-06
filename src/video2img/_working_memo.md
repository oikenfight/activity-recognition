# Frame

## やること

- 映像から任意の間隔で画像として切り出す
- 切り出した画像を任意の枚数ごとにまとめフレームを作成する
    - 前のフレームの一部を被せた形でフレームを作る
    - 例)
        - 切り出し間隔 0.5, 1フレーム10枚
        - フレーム1: 0 ~ 5 秒
        - フレーム2: 2.5 ~ 7.5 秒
        - フレーム3: 5 ~ 10 秒
- 切り出した画像をフレームごとにディレクトリへ保存

## Stair Action データセットの構成

- アクション番号対応表: 'actionlist.csv'
- ファイル命名規則
    - {action}/{アクション番号}_{ファイル番号}.mp4
    - action: アクション内容
    - アクション番号: 「a001 ~ a100」、各アクションごとに統一して振られている模様
    - ファイル番号: 「0001C ~ 」、各アクションごとで通し番号で開始番号・終了番号は不規則

```angular2html
path/to/base_data/ 
    {action}/
        {アクション番号}_{ファイル番号}.mp4
    ...
    ...
```

## 出力

```angular2html
path/to/base_output/
    {action}/
        001/
        002/
        ...
    ...
```

## 要件

- sh で実装
- mp4 ファイルを引数で受け取る
- actionlist を参照して保存先を指定する（もしくは引数で指定）
- 映像から画像を切り出す間隔は引数で受け取る
- 1フレームの長さは引数で受け取る
- 前のフレームの重複時間は引数で受ける


## 実行方法

```angular2html
$ docker-compose run --rm python src/frames/mp4_convert_batch.py
```

- 注意点
    - 環境
        - python3/ffmpeg
    - データの置き場所、保存先のディレクトリを指定・確認する
        - ファイルの置き場所を docker-compose の volume で指定
        - mp4_convert_batch で、pythonコンテナから見たデータのパスを指定
        - パス指定箇所
            - BASE_DATA
            - BASE_OUTPUT
        - base_data の置き方は上記の通り（start_action の action 一覧をそのまま置く）
