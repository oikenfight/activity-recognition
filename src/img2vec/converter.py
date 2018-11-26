import numpy as np
from PIL import Image
import chainer


class Converter:
    """
    画像を読み込んでベクトルに変換するクラス
    データの構造上 chainer の便利クラス使うとやりづらいので、あえての素の Pillow
    """
    # TODO: 余裕があったらノイズ加えるとかやりたいね。
    # TODO: VGG の学習に使うには 224 x 224 にする必要がある。時間ある時に video2img でサイズやり直す。
    WIDTH = 224
    HEIGHT = 224
    CONVERT_TYPE = 'RGB'

    def __init__(self):
        pass

    def main(self, path):
        """
        :param str path:
        :return list: list of (self.HEIGHT, self.WIDTH, [b, g, r])
        """
        img = Image.open(path)
        img = self._paste_background(img)
        # see https://github.com/chainer/chainer/blob/v5.0.0/chainer/links/model/vision/vgg.py#L466
        x = chainer.links.model.vision.vgg.prepare(img)

        # # 画像保存
        # img = x.transpose((1, 2, 0))
        # img = Image.fromarray(np.uint8(img))
        # img.save('./test.jpg')
        return x

    @staticmethod
    def _paste_background(img: Image):
        """
        画像のアスペクト比を変更しないよう、黒で余白を追加して正方形に変換
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


if __name__ == "__main__":
    # setup
    # params
    path = '/converted_data/20181106_043229/brushing_teeth/a017-0116C/0010.jpg'

    # execute
    converter_instance = Converter()
    converter_instance.main(path)
