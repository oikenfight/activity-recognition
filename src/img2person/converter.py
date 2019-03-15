import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainer import cuda
from PIL import Image
import numpy as np
from pprint import pprint as pp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()


class Converter:
    """
    YOLO v3 を使って時系列画像に変換されたデータから人物検出を行う。
    人物検出後、その部分のみを切り出してリサイズして画像を保存する。
    複数人写っている場合は、複数をまとめて切り出す。
    """
    WIDTH = 224
    HEIGHT = 224
    CONVERT_TYPE = 'RGB'
    GPU_DEVICE = 1
    PERSON_LABEL = 14

    def __init__(self):
        self.xp = np
        self.model = None
        self._prepare()

    def _prepare(self):
        self._set_model()
        self._to_gpu()

    def _set_model(self):
        self.model = YOLOv3(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='voc0712'
        )

    def _to_gpu(self):
        print('>>> use gpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        from random import randint
        a = randint(0, 1)
        if a == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        if self.GPU_DEVICE >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu()

    def main(self, input_path: str, output_path: str):
        """
        :param input_path:
        :return:
        """
        pilimg = Image.open(input_path).convert(self.CONVERT_TYPE)

        # resize （計算早くなるように少しだけど小さくしておく）
        pilimg = self._init_resize_img(pilimg)

        # print(type(pilimg))
        # pilimg.save('./tmp_test_original.jpg')

        original_width, original_height, = pilimg.size

        # 人物の座標（左上 x, y）(右下 x, y) を取得
        img = self._convert_pilimg_for_chainercv(pilimg)
        person_range = self._detect_person_range(img, original_width, original_height)

        # 画像をトリミング
        crop_img = self._crop_img(pilimg, person_range)
        # crop_img.save('./tmp_test_crop_.jpg')

        # アスペクト比を維持したまま、余白を追加し正方形にする
        pasted_img = self._paste_background(crop_img)
        # pasted_img.save('./tmp_test_pasted_img_.jpg')

        # 画像をリサイズ
        resized_image = self._resize_img(pasted_img)
        resized_image.save(output_path)

    def _init_resize_img(self, img: Image):
        """
        計算しやすいよう適当な大きさにリサイズ
        :param Image img:
        :return:
        """
        original_width, original_height, = img.size
        return img.resize((int(original_width / 2), int(original_height / 2)), Image.LANCZOS)

    def _convert_pilimg_for_chainercv(self, pilimg):
        """
        Pillow で読み込んだ画像(RGBのカラー画像とする)を、ChainerCVが扱える形式に変換
        """
        # img = np.asarray(pilimg, dtype=np.float32)
        img = np.asarray(pilimg, dtype=np.float32)
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

    def _detect_person_range(self, img: np.ndarray, original_width: int, original_height: int):
        """
         YOLO v3 を使って人物検出行い、範囲を特定する
        :return:
        """
        bboxes, labels, scores = self.model.predict([img])
        bboxes = self.xp.asarray(bboxes)
        scores = self.xp.asarray(scores)
        bbox, label, score = bboxes[0], labels[0], scores[0]

        # 人物を検出できているか
        if self.PERSON_LABEL in label:
            # （人物を複数検出する場合もある）
            person_indexes = np.where(label == self.PERSON_LABEL)[0]
        else:
            raise Exception('人物を検出できませんでした。')

        # 検出精度が曖昧な場合は除去
        if self.xp.any(score[person_indexes][:] < 0.40):
            # print(score[person_indexes][:])
            raise Exception('人物の検出結果が曖昧です。')

        # yolo が検出したやつより、少しだけ大きめに切り出す
        bbox[person_indexes, :2] -= 15   # 左上座標を検出範囲より少し大きく
        bbox[person_indexes, 2:] += 15   # 右下座標を検出範囲より少し大きく

        # 人物と特定された範囲のうち、各座標毎に最も大きい値を選択（）
        left_top_y = float(bbox[person_indexes, 0].min())
        left_top_x = float(bbox[person_indexes, 1].min())
        right_bottom_y = float(bbox[person_indexes, 2].max())
        right_bottom_x = float(bbox[person_indexes, 3].max())

        left_top_y = left_top_y if left_top_y >= 0 else 0
        left_top_x = left_top_x if left_top_x >= 0 else 0
        right_bottom_y = right_bottom_y if right_bottom_y <= original_height else original_height
        right_bottom_x = right_bottom_x if right_bottom_x <= original_width else original_width

        # 左上 x, y, 右下 x, y
        return left_top_x, left_top_y, right_bottom_x, right_bottom_y

    @staticmethod
    def _crop_img(img: Image, person_range):
        """
        検出した人物の範囲（座標）に従ってトリミング
        :param Image img:
        :param tuple person_range:
        :return:
        """
        # print(person_range)
        return img.crop(person_range)

    @staticmethod
    def _paste_background(img: Image):
        """
        トリミングした画像のアスペクト比を変更しないよう、黒で余白を追加して正方形に変換
        :param Image img:
        :return:
        """
        background_color = (0, 0, 0)
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), background_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), background_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

    def _resize_img(self, img: Image):
        """
        指定の大きさにリサイズ
        :param Image img:
        :return:
        """
        return img.resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)


if __name__ == "__main__":
    # setup
    # params
    # input_path = '/images_data/20181204_013516/fighting/a094-0209C/0001.jpg'

    # data = [
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0001.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0002.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0003.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0004.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0005.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0006.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0007.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0008.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0009.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0010.jpg',
    #     '/images_data/20181229_120953/doing_paper-rock-scissors/a093-0030C/0011.jpg',
    # ]

    data = [
        './tmp/a094-0280C/0001.jpg',
        './tmp/a094-0280C/0002.jpg',
        './tmp/a094-0280C/0003.jpg',
        './tmp/a094-0280C/0004.jpg',
        './tmp/a094-0280C/0005.jpg',
    ]

    # execute
    converter_instance = Converter()

    for i, path in enumerate(data):
        input_path = path
        output_path = './tmp_test' + str(i) + '.jpg'
        converter_instance.main(input_path, output_path)
