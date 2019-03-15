import json
from pprint import pprint as pp

JSON_PATH = './outputs/20190109_185707/results.json'


class ResultContainer:
    JSON_PATH = ''

    def __init__(self, data):
        self.data = data

    def get_result_of_action(self, action_name: str):
        result = self.data[action_name]
        print(result)
        return result

    def get_whole_accuracy(self):
        length = len(self.data)
        cnn_count, lrcn_count, person_count = 0, 0, 0
        cnn_correct, lrcn_correct, person_correct = 0, 0, 0

        for i, result in self.data.items():
            cnn_count += result['cnn_count']
            lrcn_count += result['lrcn_count']
            # person_count += result['person_count']

            cnn_correct += result['cnn_correct']
            lrcn_correct += result['lrcn_correct']
            # person_correct += result['person_correct']

        cnn_acc = cnn_correct / cnn_count
        lrcn_acc = lrcn_correct / lrcn_count
        # person_acc = person_correct_count / person_count

        print(cnn_count)
        print(lrcn_count)

        print(cnn_acc)
        print(lrcn_acc)
        # print(person_acc)

        # return cnn_acc, lrcn_acc, person_acc
        return cnn_acc, lrcn_acc


if __name__ == '__main__':
    JSON_PATH = './outputs/20190109_223705/results.json'

    f = open(JSON_PATH, 'r')
    results = json.load(f)

    pp(results)

    ResultContainer.JSON_PATH = JSON_PATH
    result_container_instance = ResultContainer(results)

    result_container_instance.get_whole_accuracy()


