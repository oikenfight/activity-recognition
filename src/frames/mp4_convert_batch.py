import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


if __name__ == '__main__':
    BASE = './src/frames'
    # コンテナからデータを見れるように deepstation のデータ保管場所を docker-compose にてマウントしてるから、その場所。
    BASE_INPUT = '/data/STAIR-actions/stair_action'
    BASE_OUTPUT = '/frame_data'
    FPS = 2

    FileManager.BASE_DIR = BASE_INPUT
    file_manager = FileManager()

    # data/以下の全ファイル（リスト型）を拾う （{action} > [{file_name}, ...]）
    all_file_lists = file_manager.all_file_lists()
    num = len(all_file_lists)

    for i, file_list in enumerate(all_file_lists):
        file_list[1] = file_list[1].split('.')[0]   # 拡張子を覗いたものをディレクトリ名にする
        path = BASE_OUTPUT + '/' + '/'.join(file_list)
        # flame_data/{action}/{file_name} ディレクトリを作成
        if not os.path.exists(path):
            os.makedirs(path)

        # 各ファイルに対して ./mp4_to_jpt_frame.sh を実行してフレームを生成
        # 実行コマンド: ./mp4_to_jpg.sh {path/to/script/dir} {path/to/input/dir} {action_name} {file_name（拡張子なし）} {path/to/output/dir} {fps}
        command = "%s/mp4_to_jpg.sh %s %s %s %s %s" % (BASE, BASE_INPUT, file_list[0], file_list[1], BASE_OUTPUT, str(FPS))
        status = os.system(command)

        progress = "%s/%s %s" % (str(i), str(num), command)
        if status != 0:
            progress += " some error occurred!!"
        print(progress)
