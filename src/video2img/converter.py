import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Converter:
    FPS = 2
    WIDTH = 300
    HEIGHT = 300

    def __init__(self):
        pass

    def main(self, input_path, output_dir):
        print()
        print("<<< Converter class: convert %s into %s as jpg >>>" % (input_path, output_dir))

        if not os.path.isfile(input_path):
            raise InputFileNotFoundError

        if not os.path.exists(output_dir):
            print('>>> create output directory.')
            os.makedirs(output_dir)

        # 横サイズを 300 にして、縦はアスペクト比を変更しないように自動スケールする
        size_option = 'scale=%s:-1' % self.WIDTH

        print('>>> execute ffmpeg command.')
        convert_command = "ffmpeg -i " + input_path + " -f image2 fps=" + str(self.FPS)\
                          + ' -vf ' + size_option + " " + output_dir + "%04d.jpg"
        print(convert_command)
        result_status = os.system(convert_command)

        if result_status != 0:
            raise FfmpegExecuteError
        return True


class FfmpegExecuteError(Exception):
    def __str__(self):
        return "入力されたファイルを jpg に適切に変換できませんでした。"


class InputFileNotFoundError(Exception):
    def __str__(self):
        return "入力されたファイルを見つけることができませんでした。"


if __name__ == '__main__':
    #
    # Example
    #

    # setup Converter
    Converter.FPS = 2

    # Test input file path
    input_path = '/stair_action/washing_hands/a007-0477C.mp4'
    output_dir = './tmp/'

    converter = Converter()
    converter.main(input_path, output_dir)

