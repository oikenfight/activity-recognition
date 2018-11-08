import numpy as np
from PIL import Image


class Converter:
    """
    画像を読み込んでベクトルに変換するクラス
    データの構造上 chainer の便利クラス使うとやりづらいので、あえての素の Pillow
    """

    WIDTH = 240
    HEIGHT = 180
    CONVERT_TYPE = 'RGB'

    def __init__(self):
        pass

    def main(self, path):
        """
        :param str path:
        :return list: list of (270, 360, 3)   # WIDTH, HEIGHT, CONVERT_TYPE による
        """
        img = Image.open(path)
        img = self._transform(img)
        pixels = list(img.getdata())
        pixels = [pixels[i * self.WIDTH:(i + 1) * self.WIDTH] for i in range(self.HEIGHT)]
        return pixels

    def _transform(self, img: Image):
        """
        前処理色々やる。
        :param img:
        :return img:
        """
        # RGB で試すけど、細かい物体認識じゃないしグレースケールでも十分？
        img = img.convert(self.CONVERT_TYPE)
        # 画像は 1440 * 1080 で必ず来るはず。。一応
        img = img.resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        return img


if __name__ == "__main__":
    # setup
    # params
    path = '/converted_data/20180929_071816/brushing_teeth/a017-0160C/0001.jpg'

    # execute
    converter_instance = Converter()
    converter_instance.main(path)
