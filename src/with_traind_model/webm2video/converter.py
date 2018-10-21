import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Converter:
    FRAME_RATE = 30

    def main(self, input_webm_path: str, output_dir: str):
        print()
        print("<<< WebmConberter class: convert %s into %s as mp4 >>>" % (input_webm_path, output_dir))

        if not os.path.isfile(input_webm_path):
            raise InputFileNotFoundError

        if not os.path.exists(output_dir):
            print('>>> create output directory.')
            os.makedirs(output_dir)

        mp4_filename = input_webm_path.split('/')[-1].split('.')[0] + '.mp4'

        print('>>> execute ffmpeg command.')
        convert_command = "ffmpeg -i " + input_webm_path + " -r " + str(self.FRAME_RATE) + ' ' + output_dir + mp4_filename
        print(convert_command)
        result_status = os.system(convert_command)

        if result_status != 0:
            raise FfmpegExecuteError
        return True


class FfmpegExecuteError(Exception):
    def __str__(self):
        return "入力されたファイルを mp4 に適切に変換できませんでした。"


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
    input_webm_path = '../data/original-actions/actions/test2/ID0057_test2.webm'
    output_dir = './tmp/'

    converter = Converter()
    converter.main(input_webm_path, output_dir)
