import numpy as np
import sys
import os
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import img2vec


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def per_pixel_mean_subtraction(images: np.ndarray) -> np.ndarray:
        """
        expected shape is (frame_num, 8, height, width, 3)
        RGB の各ピクセルごとに平均の値を引く
        :param images:
        :return:
        """
        # ピクセルごとに平均を求める
        mean_pixel_0 = np.mean(images[:, :, :, :, 0:1])
        mean_pixel_1 = np.mean(images[:, :, :, :, 1:2])
        mean_pixel_2 = np.mean(images[:, :, :, :, 2:3])

        # ピクセルごとに平均を引く
        images[:, :, :, :, 0:1] -= mean_pixel_0
        images[:, :, :, :, 1:2] -= mean_pixel_1
        images[:, :, :, :, 2:3] -= mean_pixel_2

        return images

    @staticmethod
    def color_scaling(images: np.ndarray) -> np.ndarray:
        """
        :param images:
        :return:
        """
        return images / 255

    def random_crop(self, images: np.ndarray) -> np.ndarray:
        """
        与えられた画像を、適当なサイズにリサイズし、ランダムにずらしながら取り出す
        :param images:
        :return:
        """


    def horizontal_flip(self, frame: np.ndarray) -> np.ndarray:
        """
        フレーム全体をランダムで水平方向に反転する
        :param frame:
        :return:
        """
        # 0 or 1 の乱数
        mirrorring = np.random.randint(0, 2)
        if mirrorring:
            mirrord_frame = []
            print('mirroring')
            for image in frame:
                mirrord_frame.append(image[::-1])
        return frame


if __name__ == '__main__':
    path = '/converted_data/20180929_071816/brushing_teeth/a017-0116C/0001.jpg'

    img2vec_converter_instance = img2vec.converter.Converter()
    vec = img2vec_converter_instance.main(path)

    images = np.array([[vec]]).astype(np.float32)
    preprocessor_instance = Preprocessor()
    images = preprocessor_instance.per_pixel_mean_subtraction(images)
    images[0] = preprocessor_instance.horizontal_flip(images[0])

    pilImg = Image.fromarray(np.uint8(images[0, 0]))
    pilImg.save('./test.png')

