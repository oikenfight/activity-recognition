import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Converter:
    FPS = 2

    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir

    def main(self):
        print()
        print("<<< Converter class: convert %s into %s as jpg >>>" % (self.input_path, self.output_dir))

        if not os.path.isfile(self.input_path):
            raise InputFileNotFoundError

        if not os.path.exists(self.output_dir):
            print('>>> create output directory.')
            os.makedirs(self.output_dir)

        print('>>> execute ffmpeg command.')
        convert_command = "ffmpeg -i "+self.input_path+" -f image2 -vf fps="+str(self.FPS)+' '+self.output_dir+"%04d.jpg"
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
    input_path = '../data/STAIR-actions/stair_action/brushing_teeth/a017-0116C.mp4'
    output_dir = './tmp/'

    converter = Converter(input_path, output_dir)
    converter.main()

