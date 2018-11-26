import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()


class Converter:
    """
    YOLO v3 を使って時系列画像に変換されたデータから人物検出を行う。
    人物検出後、その部分のみを切り出してリサイズして画像を保存する。
    """
    DEFAULT_WIDTH = 300
    DEFAULT_HEIGHT = 225
    WIDTH = 224
    HEIGHT = 224
    CONVERT_TYPE = 'RGB'
    GPU_DEVICE = 0
    PERSON_LABEL = 14

    def __init__(self):
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
        if self.GPU_DEVICE >= 0:
            chainer.cuda.get_device_from_id(self.GPU_DEVICE).use()
            self.model.to_gpu(self.GPU_DEVICE)

    def main(self, input_path: str, output_path: str):
        """
        :param input_path:
        :return:
        """
        pilimg = Image.open(input_path).convert(self.CONVERT_TYPE)
        pilimg.save('test_original.jpg')

        # 人物の座標（左上 x, y）(右下 x, y) を取得
        img = self._convert_pilimg_for_chainercv(pilimg)
        img = np.asarray(img)
        person_range = self._detect_person_range(img)

        # 画像をトリミング
        crop_img = self._crop_img(pilimg, person_range)
        crop_img.save('./test_crop.jpg')

        # アスペクト比を維持したまま、余白を追加し正方形にする
        pasted_img = self._paste_background(crop_img)
        pasted_img.save('./test_pasted_img.jpg')

        # 画像をリサイズ
        resized_image = self._resize_img(pasted_img)
        resized_image.save(output_path)

    @staticmethod
    def _convert_pilimg_for_chainercv(pilimg):
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
            person_index = np.where(label == self.PERSON_LABEL)[0][0]
        else:
            raise Exception('人物を検出できませんでした。')
        # 検出精度が曖昧な場合は除去
        if score[person_index] < 0.75:
            raise Exception('人物の検出結果が曖昧です。')

        # yolo が検出したやつより、少しだけ大きめに切り出す
        bbox[person_index][:2] -= 15
        bbox[person_index][2:] += 15
        bbox[person_index][bbox[person_index] < 0] = 0
        bbox[person_index][3] = bbox[person_index][3] if bbox[person_index][3] < self.DEFAULT_WIDTH else self.DEFAULT_WIDTH
        bbox[person_index][2] = bbox[person_index][2] if bbox[person_index][2] < self.DEFAULT_HEIGHT else self.DEFAULT_HEIGHT

        # 左上 x, y, 右下 x, y
        return bbox[person_index][1], bbox[person_index][0], bbox[person_index][3], bbox[person_index][2]

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
    input_path = '/converted_data/20181106_043229/drinking/a001-0485C/0001.jpg'
    output_path = './test.jpg'

    # execute
    converter_instance = Converter()
    converter_instance.main(input_path, output_path)
