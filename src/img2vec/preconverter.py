import numpy as np
from PIL import Image
import chainer


class PreConverter:
    """
    画像を読み込んでリサイズするクラス
    学習中にリサイズしてるとデータの読み込みに時間がかかるから、あらかじめ全画像ファイルをリサイズしておく！
    """
    # TODO: 余裕があったらノイズ加えるとかやりたいね。
    # TODO: VGG の学習に使うには 224 x 224 にする必要がある。時間ある時に video2img でサイズやり直す。
    WIDTH = 224
    HEIGHT = 224
    CONVERT_TYPE = 'RGB'

    def __init__(self):
        pass

    def main(self, input_path: str, output_path: str):
        """
        :param str input_path:
        :param str output_path:
        """
        img = Image.open(input_path)
        img = self._paste_background(img)

        # 画像保存
        img = Image.fromarray(np.uint8(img))
        img.save(output_path)

    @staticmethod
    def _paste_background(img: np.ndarray):
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
    path = '/images_data/20181204_013516/brushing_teeth/a017-0116C/0010.jpg'

    # execute
    converter_instance = PreConverter()
    converter_instance.main(path)
