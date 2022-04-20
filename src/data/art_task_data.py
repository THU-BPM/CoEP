import pandas as pd
import random
import json
from src.callback.progressbar import ProgressBar
from src.common.tools import save_pickle, logger


class TaskData(object):
    def __init__(self, raw_data_path, preprocessor=None, is_train=True):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train

    def read_data(self):
        """
        :@param relations: list of str
        :return: list of targets and sentences
        """
        data = pd.read_csv(self.data_path)

        targets_list = data["ob2"].values
        sentences_list = data["ob1"].values
        relations_list = data["hyp"].values
        return list(targets_list), list(sentences_list), list(relations_list)

    def save_data(self, X, y, c=None, shuffle=True, seed=None, data_name=None, data_dir=None, data_split=None):
        """
        save data in pkl format
        :param data_split: train or valid or test
        :param X:
        :param y:
        :param shuffle:
        :param seed:
        :param data_name:
        :param data_dir:
        :return:
        """
        pbar = ProgressBar(n_total=len(X), desc='save')
        logger.info('save data in pkl for training')
        if c:
            data = []
            for step, (data_x, data_y, data_c) in enumerate(zip(X, y, c)):
                data.append((data_x, data_y, data_c))
                pbar(step=step)
            del X, y, c
        else:
            data = []
            for step, (data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y, None))
                pbar(step=step)
            del X, y
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        file_path = data_dir / f"{data_name}.{data_split}.pkl"
        save_pickle(data=data, file_path=file_path)
