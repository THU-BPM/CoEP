import pandas as pd
import random
from src.callback.progressbar import ProgressBar
from src.common.tools import save_pickle, logger


class TaskData(object):
    def __init__(self, raw_data_path, preprocessor=None, is_train=True):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train

    def read_data(self):
        """
        :return: list of targets and sentences
        targets: the same as stories,
        stories: five sentences,
        chars: five None token
        """
        logger.info(f'context_motiv_task read data from {self.data_path}')
        logger.info(f'is_train tag: {self.is_train}')
        targets, stories, chars = [], [], []
        data = pd.read_csv(self.data_path)
        length = len(data.values)
        logger.info(f'number of data in {self.data_path}: {length}')
        for i in range(length):
            context = str(data.loc[i, 'context'])
            context_list = context.split('\t')[1:]
            assert len(context_list) == 5
            if self.preprocessor:
                context_list = [self.preprocessor(context) for context in context_list]

            targets.append(context_list[1:])
            stories.append(context_list[:-1])
            chars.append([None] * 4)
        return targets, stories, chars

    def save_data(self, X, y, c=None, shuffle=False, seed=None, data_name=None, data_dir=None, data_split=None):
        """
        save data in pkl format
        :param c: List(str)
        :param data_split: train or valid or test
        :param X: List(list(str))
        :param y: List(list(str))
        :param shuffle:
        :param seed:
        :param data_name:
        :param data_dir:
        :return:
        """
        pbar = ProgressBar(n_total=len(X), desc='save')
        logger.info(f'save data in pkl for {data_split}')
        logger.info(f'number of data: {len(X)}')
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