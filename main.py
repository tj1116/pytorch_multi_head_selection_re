import os
import json
import time
import argparse

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD

from lib.preprocessings import Chinese_selection_preprocessing, NYT_bert_selection_preprocessing
from lib.preprocessings import NYT_selection_preprocessing, NYT11_bert_selection_preprocessing
from lib.preprocessings import NYT10_bert_selection_preprocessing
from lib.dataloaders import Selection_bert_loader
from lib.dataloaders import Selection_Dataset, Selection_loader, Selection_Nyt_Dataset, Selection_bert_Nyt_Dataset
from lib.metrics import F1_triplet
from lib.models import MultiHeadSelection
from lib.config import Hyper

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='chinese_selection_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation')
args = parser.parse_args()

class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu

        if self.hyper.is_bert == 'ERNIE':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)

        elif self.hyper.is_bert == "bert_bilstem_crf":
            self.preprocessor = NYT_selection_preprocessing(self.hyper)

        elif self.hyper.is_bert == "nyt_bert_tokenizer":
            self.preprocessor = NYT_bert_selection_preprocessing(self.hyper)

        elif self.hyper.is_bert == "nyt11_bert_tokenizer":
            self.preprocessor = NYT11_bert_selection_preprocessing(self.hyper)

        elif self.hyper.is_bert == "nyt10_bert_tokenizer":
            self.preprocessor = NYT10_bert_selection_preprocessing(self.hyper)


        self.metrics = F1_triplet()

    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.01)
        }
        return m[name]

    def init_MultiHeadSelection(self):
        self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)
        self.optimizer = self._optimizer(self.hyper.optimizer, self.model)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        self.preprocessor.gen_ERNIE_vocab()
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self.init_MultiHeadSelection()

            #self.load_lastest_models()
            #self.load_model(40)
            self.train()
        elif mode == 'evaluation':
            self.init_MultiHeadSelection()
            #self.load_model(epoch=self.hyper.evaluation_epoch)
            self.load_lastest_models()

            #self.load_model(0)
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def get_lastest_model_dir(self, model_dir: str):
        file_new = ''
        lists = os.listdir(model_dir)
        if len(lists) != 0:
            lists.sort(key=lambda fn: os.path.getmtime(model_dir + "/" + fn))
            file_new = os.path.join(model_dir, lists[-1])
        return file_new

    def load_lastest_models(self):
        model_dir = self.get_lastest_model_dir(self.model_dir)
        if model_dir != '':
            self.model.load_state_dict(torch.load(model_dir))
        return None

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    def evaluation(self):
        if self.hyper.is_bert == "nyt_bert_tokenizer" or \
                self.hyper.is_bert == "nyt11_bert_tokenizer" or \
                self.hyper.is_bert == "nyt10_bert_tokenizer":
            dev_set = Selection_bert_Nyt_Dataset(self.hyper, self.hyper.dev)
            loader  = Selection_bert_loader(dev_set, batch_size=100, pin_memory=True)

        elif self.hyper.is_bert == "bert_bilstem_crf":
            dev_set = Selection_Nyt_Dataset(self.hyper, self.hyper.dev)
            loader  = Selection_loader(dev_set, batch_size=100, pin_memory=True)

        else:
            dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
            loader = Selection_loader(dev_set, batch_size=100, pin_memory=True)


        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                #self.metrics(output['selection_triplets'], output['spo_gold'])
                self.metrics(output, output)

            result = self.metrics.get_metric()
            print(', '.join([
                "%s: %.4f" % (name, value)
                for name, value in result.items() if not name.startswith("_")
            ]) + " ||")

    def train(self):

        if self.hyper.is_bert == "bert_bilstem_crf":
            train_set = Selection_Nyt_Dataset(self.hyper, self.hyper.train)
            loader = Selection_loader(train_set, batch_size=100, pin_memory=True)

        elif self.hyper.is_bert == "nyt_bert_tokenizer" or \
                self.hyper.is_bert == "nyt11_bert_tokenizer" or \
                self.hyper.is_bert == "nyt10_bert_tokenizer":
            train_set = Selection_bert_Nyt_Dataset(self.hyper, self.hyper.train)
            loader = Selection_bert_loader(train_set, batch_size=100, pin_memory=True)

        else:
            train_set = Selection_Dataset(self.hyper, self.hyper.train)
            loader = Selection_loader(train_set, batch_size=100, pin_memory=True)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)
            if epoch > 3:
                self.evaluation()

            #if epoch >= 6:
            #    self.evaluation()

            '''
            if epoch % self.hyper.print_epoch == 0:
                self.evaluation()
            '''

if __name__ == "__main__":
    y = 'NYT_bert_selection_re'

    #x = 'preprocessing'

    x = 'train'
    #x = 'evaluation'
    '''
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
    
    '''

    config = Runner(exp_name=y)
    config.run(mode=x)
