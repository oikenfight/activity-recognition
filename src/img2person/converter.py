import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox
from PIL import Image
import numpy as np
import cupy

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
    GPU_DEVICE = 0
    PERSON_LABEL = 14

    def __init__(self):
        self.model = None
        self.original_height = 270  # default
        self.original_width = 360  # default
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
        if self.GPU_DEVICE >= 0:
            chainer.cuda.get_device_from_id(self.GPU_DEVICE).use()
            self.model.to_gpu(self.GPU_DEVICE)

    def main(self, input_path: str, output_path: str):
        """
        :param input_path:
        :return:
        """
        pilimg = Image.open(input_path).convert(self.CONVERT_TYPE)

        # set original image size
        self.original_width, self.original_height, = pilimg.size

        # pilimg.save('test_original.jpg')

        # 人物の座標（左上 x, y）(右下 x, y) を取得
        img = self._convert_pilimg_for_chainercv(pilimg)
        img = np.asarray(img)
        person_range = self._detect_person_range(img)

        # 画像をトリミング
        crop_img = self._crop_img(pilimg, person_range)
        # crop_img.save('./test_crop.jpg')

        # アスペクト比を維持したまま、余白を追加し正方形にする
        pasted_img = self._paste_background(crop_img)
        # pasted_img.save('./test_pasted_img.jpg')

        # 画像をリサイズ
        resized_image = self._resize_img(pasted_img)
        resized_image.save(output_path)

    def _convert_pilimg_for_chainercv(self, pilimg):
        """
        Pillow で読み込んだ画像(RGBのカラー画像とする)を、ChainerCVが扱える形式に変換
        """
        img = np.asarray(pilimg, dtype=np.float32)
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

    def _detect_person_range(self, img: np.ndarray):
        """
         YOLO v3 を使って人物検出行い、範囲を特定する
        :return:
        """
        bboxes, labels, scores = self.model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        # 人物を検出できているか
        if self.PERSON_LABEL in label:
            # （人物を複数検出する場合もある）
            person_indexes = np.where(label == self.PERSON_LABEL)[0]
        else:
            raise Exception('人物を検出できませんでした。')

        # 検出精度が曖昧な場合は除去
        if np.any(score[person_indexes][:] < 0.50):
            print(score[person_indexes][:])
            raise Exception('人物の検出結果が曖昧です。')

        # yolo が検出したやつより、少しだけ大きめに切り出す
        bbox[person_indexes][:, 2] -= 15   # 左上座標を 15 大きく
        bbox[person_indexes][:, 2:] += 15   # 右下座標を 15 大きく

        # 補正された人物範囲の値のうち、最大値（最小値）を超えている場合は、最大値（最小値）に修正
        bbox[person_indexes][:, 0][bbox[person_indexes][:, 0] < 0] = 0    # 左上 y 座標が 0 より小さい場合 0 に補正
        bbox[person_indexes][:, 1][bbox[person_indexes][:, 1] < 0] = 0    # 左上 x 座標が 0 より小さい場合 0 に補正
        bbox[person_indexes][:, 2][bbox[person_indexes][:, 2] < self.original_height] = self.original_height    # 右下 y 座標が max より大きい場合 max に補正
        bbox[person_indexes][:, 2][bbox[person_indexes][:, 2] < self.original_width] = self.original_width    # 右下 x 座標が max より大きい場合 max に補正

        # 人物と特定された範囲のうち、各座標毎に最も大きい値を選択（）
        left_top_y = bbox[person_indexes][:, 0].min()
        left_top_x = bbox[person_indexes][:, 1].min()
        right_bottom_y = bbox[person_indexes][:, 2].max()
        right_bottom_x = bbox[person_indexes][:, 3].max()

        # print((left_top_x, left_top_y, right_bottom_x, right_bottom_y))
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
    input_path = './a072-0587C/0001.jpg'
    output_path = './test.jpg'

    # execute
    converter_instance = Converter()
    converter_instance.main(input_path, output_path)
