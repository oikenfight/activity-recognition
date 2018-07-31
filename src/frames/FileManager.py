import os


class FileManager:
    # 使用場所ににって適切なディレクトリを指定してからインスタンス化する
    BASE_DIR = '.'

    def __init__(self):
        self.current_path = self.BASE_DIR
        self.current_path_list = []
        self.all_paths = []
        self.all_path_lists = []

    def all_dirs(self):
        self._search_dirs()
        all_dirs = self.all_paths
        self.__reset_all()
        return all_dirs

    def all_files(self):
        self._search_files()
        all_path_lists = self.all_paths
        self.__reset_all()
        return all_path_lists

    def all_dir_lists(self):
        self._search_dirs()
        all_dir_lists = self.all_path_lists
        self.__reset_all()
        return all_dir_lists

    def all_path_lists(self):
        self._search_files()
        all_path_lists = self.all_paths
        self.__reset_all()
        return all_path_lists

    def _update(self, item: str):
        # 現在のパスを更新
        self.current_path += '/' + item
        self.current_path_list += [item]

    def _append_dir_path(self):
        # 現在のパスを追加
        self.all_paths += [self.current_path]
        self.all_path_lists += [self.current_path_list]

    def _append_file_path(self, files):
        for file in files:
            # 現在のディレクトリのパス+ファイル名を追加
            self.all_paths += [self.current_path + '/' + file]
            self.all_path_lists += [self.current_path_list + [file]]

    def _reset_current(self, current_path, current_path_list):
        # 直前に更新したパスをもとに戻す
        self.current_path = current_path
        self.current_path_list = current_path_list

    def __reset_all(self):
        self.__init__()

    def _search_dirs(self):
        # インスタンス変数は更新されていくため、データを保持しておく
        current_path = self.current_path
        current_path_list = self.current_path_list
        # ディレクトリ一覧を取得
        files = os.listdir(current_path)
        # ディレクトリ一覧
        dirs = [f for f in files if os.path.isdir(os.path.join(current_path, f))]

        if dirs:
            for item in dirs:
                self._update(item)    # 現在のディレクトリを見つけたディレクトに更新
                self._search_dirs()  # 見つけたディレクトリの下を探索
                self._reset_current(current_path, current_path_list)  # 前の状態に戻す
        else:
            # 以下にディレクトリが見つからなくなったら（ディレクトリの最下層に達したら）、現在までのパスを追加
            self._append_dir_path()

    def _search_files(self):
        # インスタンス変数は更新されていくため、データを保持しておく
        current_path = self.current_path
        current_path_list = self.current_path_list
        # ファイル + ディレクトリ一覧
        files = os.listdir(current_path)
        # ディレクトリ一覧
        dirs = [f for f in files if os.path.isdir(os.path.join(current_path, f))]
        # ファイル一覧
        files = [f for f in files if os.path.isfile(os.path.join(current_path, f))]

        # 現在のディレクトリ内のファイルのパスを追加
        self._append_file_path(files)

        # ディレクトリが存在するすれば探索
        for item in dirs:
            self._update(item)  # 現在のディレクトリを見つけたディレクトに更新
            self._search_files()  # 見つけたディレクトリの下を探索
            self._reset_current(current_path, current_path_list)  # 前の状態に戻す


if __name__ == '__main__':
    # 探索したいディレクトリを指定してからインスタンス化する
    FileManager.BASE_DIR = '/home/oikenfight/workspace/data/STAIR-actions/STAIR_Actions_v1.0/'
    file_manager = FileManager()

    print('===== all dirs ==================')
    for path in file_manager.all_dirs():
        print(path)

    print()
    print('===== all files ==================')
    for path in file_manager.all_files():
        print(path)

