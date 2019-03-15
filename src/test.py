import pickle as pkl


def _load_actions(path):
    with open(path, mode="rb") as f:
        actions = pkl.load(f)
    print('action length: ', str(len(actions)))
    print(actions)


def _load_test_data(path):
    with open(path, mode="rb") as f:
        test_data_paths = pkl.load(f)
    print('test data length: ', str(len(test_data_paths)))


if __name__ == '__main__':
    action_path = 'output/lrcn_recognition/models/20190101_063332/actions.pkl'
    test_data_path = 'output/lrcn_recognition/models/20190101_063332/test_frame_data.pkl'

    _load_actions(action_path)
    _load_test_data(test_data_path)