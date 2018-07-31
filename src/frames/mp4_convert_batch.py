import os
from FileManager import FileManager


if __name__ == '__main__':
    BASE = './src/frames'
    BASE_DATA = BASE + '/data'
    BASE_OUTPUT = BASE + '/frame_data'
    FPS = 1
    FileManager.BASE_DIR = BASE_DATA
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
        # 実行コマンド: ./mp4_to_jpg_frame.sh brushing_teeth a017-0115C 1
        # 実行コマンド: ./mp4_to_jpg_frame.sh {アクション名} {ファイル名（拡張子なし）} {fps}
        command = "%s/mp4_to_jpg_frame.sh %s %s %s" % (BASE, file_list[0], file_list[1], str(FPS))
        status = os.system(command)

        progress = "%s/%s %s" % (str(i), str(num), command)
        if status != 0:
            progress += " some error occurred!!"
        print(progress)
