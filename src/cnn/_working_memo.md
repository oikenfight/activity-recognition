# CNN

## やること

- 切り出した各画像から特徴量抽出する
- 特徴抽出結果を後で利用しやすいように階層化して pkl で出力しておく

## 出力（.pkl）

ファイル保存場所
```
src/cnn/dataset/
    {datetime}/
        {action}}.pkl,
        {action}}.pkl,
        ...
    {datetime}/
        {action}}.pkl,
        ...
    ...
    ...
```

pkl 保存形式
```
data = {
    {filename}: [
        0: [....],
        1: [....],
        ...
    ],
    {filename}: [
        0: [....],
        1: [....],
        ...
    ],
    ...
}
```

## 要件

- 全てのデータを特徴抽出する
    - 実行はアクション単位で行いループさせる
    - pkl もアクション数分出力する
    - メソッド化して任意のアクションのみ実行を可能にする
- 出力形式を満たす
- 出力ファイル名は {datetime}/{action}.pkl とする
    - action ごとに pkl が作成される
- 保存場所は一旦 src/cnn/dataset 以下とする

## 実行
- 設定項目
    - BASE: ファイルの実行場所
    - BASE_OUTPUT: 保存先（dataset/ までのpath）
    - INPUT_FILES_BASE: 切り出した画像ファイルが置かれているベースディレクトリ
    - MODEL_PATH: cafeemodel がある path

```angular2html
# set up
Features.BASE = BASE
Features.BASE_OUTPUT = OUTPUT_BASE
Features.INPUT_FILES_BASE = INPUT_FILES_BASE
Features.MODEL_PATH = MODEL_PATH

# execute
features.main()
```

- 実行コマンド
```angular2html
$ docker-compose run --rm python ./src/cnn/Features.py
```