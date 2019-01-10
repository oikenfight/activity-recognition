import pickle as pkl


class ResultContainer:
    # constant
    ACTIONS_PKL_PATH = ''
    RECOGNIZED_RESULT_PKL_PATH = ''

    def __init__(self, actions: dict, data: dict):
        self.actions = actions
        self.data = data

    def video_name_data_of(self, action_name: str) -> list:
        """
        アクション名からビデオ名一覧を取得
        :param action_name:
        :return:
        """
        video_names = []
        for video_name, result in sorted(self.data[action_name].items()):
            video_names.append(video_name)
        return video_names
    #
    # def row_data_of(self, action_name, video_name):
    #     return self.actions[action_name][video_name]

    def data_of_action(self, action_name: str) -> list:
        """
        アクション名から認識結果一覧を取得
        :param action_name:
        :return: [({max_index}, {max_value}, {actual_index}, {actual_value}), ...]
        """
        data = []
        if self.data.get(action_name):
            for video_name, results in sorted(self.data[action_name].items()):
                data.append([result for result in results])
        return data

    # def data_of_video(self, action_name: str, video_name) -> list:
    #     """
    #     ビデオ名から認識結果一覧を取得
    #     :param action_name:
    #     :param video_name:
    #     :return: [({max_index}, {max_value}, {actual_index}, {actual_value}), ...]
    #     """
    #     return self.data[action_name][video_name]


if __name__ == '__main__':
    ACTIONS_PKL_PATH = './actions.pkl'
    RECOGNIZED_RESULT_PKL_PATH = './20181223_214438_result.pkl'

    # set up
    ResultContainer.ACTIONS_PKL_PATH = ACTIONS_PKL_PATH
    ResultContainer.RECOGNIZED_RESULT_PKL_PATH = RECOGNIZED_RESULT_PKL_PATH

    # params
    with open(ACTIONS_PKL_PATH, mode="rb") as f:
        actions = pkl.load(f)
    print('action data is loaded. (length: ', str(len(actions)), ')')

    with open(RECOGNIZED_RESULT_PKL_PATH, mode="rb") as f:
        data = pkl.load(f)
    print('result data is loaded.')

    result_container_instance = ResultContainer(actions, data)
    print(result_container_instance.data_of('drinking'))

