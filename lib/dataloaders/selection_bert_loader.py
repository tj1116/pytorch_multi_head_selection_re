import os
import json
import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_pretrained_bert import BertTokenizer

class Selection_bert_Nyt_Dataset(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper.data_root

        if self.hyper.is_bert == "bert_bilstem_crf" or \
                self.hyper.is_bert == "bert_tokenizer" or \
                self.hyper.is_bert == 'nyt11_bert_tokenizer' or \
                self.hyper.is_bert == 'nyt10_bert_tokenizer':
            self.word_vocab = json.load(
                open(os.path.join(self.data_root, 'word_bert_vocab.json'), 'r'))

            self.relation_vocab = json.load(
                open(os.path.join(self.data_root, 'relations2id.json'), 'r'))
        else:
            self.word_vocab = json.load(
                open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
            self.relation_vocab = json.load(
                open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))

        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []
        self.ner_tokenizer_list = []

        vocab_path = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])
            self.ner_tokenizer_list.append(instance['ner_tokenizer'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        ner_tokenizer = self.ner_tokenizer_list[index]

        text_cut = self.tokenizer.tokenize(text)
        text_cut.insert(0, "[CLS]")

        tokens_id = self.text2tensor(text_cut)
        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text_cut, selection)

        return tokens_id, bio_id, selection_id, len(text_cut), spo, text_cut, bio, ner_tokenizer

    def __len__(self):
        return len(self.text_list)

    def text2tensor(self, text: List[str]) -> torch.tensor:

        text_len = len(text)
        '''
        for words in text:
            if words == '[CLS]':
                indexed_tokens.append(self.word_vocab[words])
                continue
            words = words.lower()
            if words in self.word_vocab:
                indexed_tokens.append(self.word_vocab[words])
            else:
                indexed_tokens.append(self.word_vocab['[UNK]'])
        '''
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)
        #indexed_tokens.extend([self.word_vocab['[PAD]']]*(self.hyper.max_text_len-text_len))
        indexed_tokens.extend([0] * (self.hyper.max_text_len - text_len))

        return torch.tensor(indexed_tokens)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.

        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] * (self.hyper.max_text_len - len(bio))) #<pad>

        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        NA = self.relation_vocab['None']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0

        return result


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, spo, text, bio

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
        self.selection_id = torch.stack(transposed_data[2], 0)

        self.length = transposed_data[3]

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6]
        self.ner_tokenizer = transposed_data[7]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_bert_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)