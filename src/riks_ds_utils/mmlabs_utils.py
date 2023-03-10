import json
import random
from copy import deepcopy


class MMLabsUtils():

    def _load_json(p):

        with open(p, 'r') as f:
            dataset = json.load(f)

        return dataset

    def split_ocr_dataset(json_file, val_ratio):

        dataset = MMLabsUtils._load_json(json_file)

        val = deepcopy(dataset)
        train = deepcopy(dataset)

        random.shuffle(dataset['data_list'])

        offset = int(val_ratio * len(dataset['data_list']))

        val_data_list = dataset['data_list'][0:offset]
        train_data_list = dataset['data_list'][offset:]

        val['data_list'] = val_data_list
        train['data_list'] = train_data_list

        return (train, val)
