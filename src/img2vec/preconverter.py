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

        # 画像をリサイズ
        resized_image = self._resize_img(img)

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
    input_paths = [
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/001.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/002.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/003.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/004.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/005.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/006.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/007.jpg',
        # '/resized_images_data/20181221_111226/putting_on_cloth/a052-0289C/008.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/001.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/002.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/003.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/004.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/005.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/006.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/007.jpg',
        '/images_data/20181204_013516//putting_on_cloth/a052-0289C/008.jpg',
    ]
    output_paths = [
        './tmp/a052-0289C_result/001.jpg',
        './tmp/a052-0289C_result/002.jpg',
        './tmp/a052-0289C_result/003.jpg',
        './tmp/a052-0289C_result/004.jpg',
        './tmp/a052-0289C_result/005.jpg',
        './tmp/a052-0289C_result/006.jpg',
        './tmp/a052-0289C_result/007.jpg',
        './tmp/a052-0289C_result/008.jpg',
    ]

    # execute
    converter_instance = PreConverter()

    for input_path, output_path in zip(input_paths, output_paths):
        converter_instance.main(input_path, output_path)

