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

        # img.save('./test2.jpg')

        x = chainer.links.model.vision.vgg.prepare(img)
        # # 画像保存
        # img = x.transpose((1, 2, 0))
        # img = Image.fromarray(np.uint8(img))
        # img.save('./test.jpg')
        return x

    def _transform(self, img: Image):
        """
        前処理色々やる。
        :param img:
        :return img:
        """
        img = img.convert(self.CONVERT_TYPE)
        # 画像は self.WIDTH * self.HEIGHT で必ず来るはず。。一応
        img = img.resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        return img

    def _to_bgr_pixel(self, img: Image):
        """
        :param Image img:
        :return list bgr_pixels:
        """
        bgr_pixels = []
        for y in range(self.HEIGHT):
            row = []
            for x in range(self.WIDTH):
                r, g, b = img.getpixel((x, y))

                r = (r - 123.68) / 255.0
                g = (g - 116.779) / 255.0
                b = (b - 103.939) / 255.0

                # TODO: VGG で読むには BGR の順になってないといけない。VGG 使うなら最初から openCV で読もうな。
                row.append([b, g, r])
            bgr_pixels.append(row)

        # img = np.asarray(bgr_pixels)
        # print(img.shape)
        # img = Image.fromarray(np.uint8(img))
        # img.save('./test.jpg')

        return bgr_pixels


if __name__ == "__main__":
    # setup
    # params
    path = '/converted_data/20181106_043229/brushing_teeth/a017-0116C/0010.jpg'

    # execute
    converter_instance = Converter()
    converter_instance.main(path)
