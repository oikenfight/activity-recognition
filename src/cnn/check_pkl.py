import pickle as pkl
from FileManager import FileManager


if __name__ == "__main__":
    BASE = './src/cnn'
    BASE_DATA = BASE + '/dataset/20180815_052528'
    FileManager.BASE_DIR = BASE_DATA
    file_manager = FileManager()

    all_files = file_manager.all_files()

    for file in all_files:
        print('=================================')
        print(file)
        with open(file, 'rb') as f:
            dataset = pkl.load(f)

        for folder_name, features in dataset.items():
            print('-----------------------------------------')
            print('folder name: ', str(folder_name))
            for j, feature in enumerate(features):
                print('number:', str(j))
                print(feature)
