import os
from datetime import datetime
import pickle as pkl
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from result_container import ResultContainer
from pprint import pprint as pp
import json


class Analysis:
    # set up
    CNN_RECOGNIZED_RESULT_PKL_PATH = ''
    LRCN_RECOGNIZED_RESULT_PKL_PATH = ''
    PERSON_LRCN_RECOGNIZED_RESULT_PKL_PATH = ''
    ACTIONS_PKL_PATH = ''

    def __init__(self):
        # actions
        self.actions = None
        # instance
        self.cnn_result_container_instance = None
        self.lrcn_result_container_instance = None
        self.person_lrcn_result_container_instance = None
        # recognized data
        self.cnn_result_container_data = None
        self.lrcn_result_container_data = None
        self.person_lrcn_result_container_data = None
        # others
        self.output_dir = ''
        # init
        self._prepare()

    def _prepare(self):
        self._init_actions()
        self._init_data()
        self._set_result_container()
        self._make_output_dir()

    def _init_actions(self):
        with open(self.ACTIONS_PKL_PATH, mode="rb") as f:
            self.actions = pkl.load(f)
        print('action data is loaded. (length: ', str(len(self.actions)), ')')

    def _init_data(self):
        with open(self.CNN_RECOGNIZED_RESULT_PKL_PATH, mode="rb") as f:
            self.cnn_result_container_data = pkl.load(f)

        with open(self.LRCN_RECOGNIZED_RESULT_PKL_PATH, mode="rb") as f:
            self.lrcn_result_container_data = pkl.load(f)

        with open(self.PERSON_LRCN_RECOGNIZED_RESULT_PKL_PATH, mode="rb") as f:
            self.person_lrcn_result_container_data = pkl.load(f)
        print('result data is loaded.')

    def _set_result_container(self):
        ResultContainer.ACTIONS_PKL_PATH = self.ACTIONS_PKL_PATH

        ResultContainer.RECOGNIZED_RESULT_PKL_PATH = self.CNN_RECOGNIZED_RESULT_PKL_PATH
        self.cnn_result_container_instance = ResultContainer(self.actions, self.cnn_result_container_data)

        ResultContainer.RECOGNIZED_RESULT_PKL_PATH = self.LRCN_RECOGNIZED_RESULT_PKL_PATH
        self.lrcn_result_container_instance = ResultContainer(self.actions, self.lrcn_result_container_data)

        # ResultContainer.RECOGNIZED_RESULT_PKL_PATH = self.PERSON_LRCN_RECOGNIZED_RESULT_PKL_PATH
        # self.person_lrcn_result_container_instance = ResultContainer(self.actions, self.person_lrcn_result_container_data)

    def _make_output_dir(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = 'outputs/' + current_time + '/'

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _create_graph(self, action: str, cnn_data: list, lrcn_data: list, person_data: list):
        # Figureを作成
        fig = plt.figure()
        # グリッド線の表示
        plt.style.use("ggplot")
        # FigureにAxesを１つ追加
        ax = fig.add_subplot(1, 1, 1)
        # Axesのタイトルの設定
        ax.set_title(action, fontsize=17)
        # 軸を設定
        ax.set_ylabel('relative frequency', fontsize=15)
        ax.set_xlabel('correct score', fontsize=15)
        # ヒストグラムを描画
        ax.hist([cnn_data, lrcn_data], label=['cnn', 'lrcn'], bins=20, density=True)
        ax.legend(loc='upper left', fontsize=15)
        # 画像を保存
        output_path = self.output_dir + action + '.png'
        plt.savefig(output_path)

    def _analyze_difference_by_method_of(self, data_of_action: list):
        pass

    def main(self):
        # データ集計用
        result_data_json = {}

        for label, action_name in self.actions.items():
            # results: [[({max_label}, {max_value}, {actual_label}, {actual_value}), ...], ...]
            cnn_results = self.cnn_result_container_instance.data_of_action(action_name)
            lrcn_results = self.lrcn_result_container_instance.data_of_action(action_name)
            # person_results = self.person_lrcn_result_container_instance.data_of_action(action_name)

            cnn_data, lrcn_data, person_data = [], [], []
            cnn_count, lrcn_count, person_count = 0, 0, 0
            cnn_correct, lrcn_correct, person_correct = 0, 0, 0

            for cnn_result in cnn_results:
                for val in cnn_result:
                    cnn_data += [val[3]]
                    cnn_count += 1
                    if val[0] == val[2]:
                        cnn_correct += 1
            for lrcn_result in lrcn_results:
                for val in lrcn_result:
                    lrcn_data += [val[3]]
                    lrcn_count += 1
                    if val[0] == val[2]:
                        lrcn_correct += 1
            # for person_result in person_results:
            #     for val in person_result:
            #         person_data += [val[3]]
            #         person_count += 1
            #         if val[0] == val[2]:
            #             person_correct += 1

            # for lrcn_result in lrcn_results:
            #     lrcn_data += [val[3] for val in lrcn_result]
            # for person_result in person_results:
            #     person_data += [val[3] for val in person_result]

            result_data_json[action_name] = {
                'cnn_count': cnn_count,
                'lrcn_count': lrcn_count,
                # 'person_count': person_count,
                'cnn_correct': cnn_correct,
                'lrcn_correct': cnn_correct,
                # 'person_correct': cnn_correct,
                'cnn_accuracy': cnn_correct / cnn_count if not cnn_count == 0 else 0,
                'lrcn_accuracy': lrcn_correct / lrcn_count if not lrcn_count == 0 else 0,
                # 'person_accuracy': person_correct / person_correct if not person == 0 else 0,
            }

            # create graph
            self._create_graph(action_name, cnn_data, lrcn_data, person_data)
        self._dump_to(result_data_json)

    def _dump_to(self, result_data_json: dict):
        output_path = self.output_dir + 'results.json'
        f = open(output_path, 'w')
        json.dump(result_data_json, f)


if __name__ == '__main__':
    # set up
    Analysis.ACTIONS_PKL_PATH = './actions.pkl'
    Analysis.CNN_RECOGNIZED_RESULT_PKL_PATH = './20190107_202636_cnn_result.pkl'
    Analysis.LRCN_RECOGNIZED_RESULT_PKL_PATH = './20190108_082012_lrcn_result.pkl'
    Analysis.PERSON_LRCN_RECOGNIZED_RESULT_PKL_PATH = './20190107_202636_cnn_result.pkl'

    analysis_instance = Analysis()
    analysis_instance.main()
