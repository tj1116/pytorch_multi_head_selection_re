import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property

#from lib.config import Hyper

import nltk
#nltk.download('punkt')

class NYT_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)


        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

        '''
        self.relation_vocab_path = os.path.join('../../data/nyt/multi_head_selection/',
                                                hyper.relation_vocab)
        '''

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        #result = {'<pad>': 5, 'B': 0, 'I': 1, 'O': 2, '<start>': 3, '<stop>': 4}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        return None

    def gen_vocab(self, min_freq: int):
        return None

    def gen_ERNIE_vocab(self):
        return None

    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text        = instance['sentText']
        sfEntityMentions = instance['sfEntityMentions']
        text_cut   = nltk.word_tokenize(text)
        bio = None
        selection = None

        if 'relationMentions' in instance:
            spo_list = instance['relationMentions']

            if len(text_cut) > self.hyper.max_text_len - 1:
                return None

            if not self._check_valid(text, spo_list):
                return None
            spo_list = [{
                'predicate': spo['label'],
                'object': spo['em2Text'],
                'subject': spo['em1Text']
            } for spo in spo_list]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text_cut, sfEntityMentions, entities)
            selection = self.spo_to_selection(sfEntityMentions, spo_list)

        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection
        }
        return json.dumps(result, ensure_ascii=False)

    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)

        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)

    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        '''
        if len(text) > self.hyper.max_text_len:
            return False
        '''
        for t in spo_list:
            if t['em2Text'] not in text or t['em1Text'] not in text:
                return False
        return True

    def spo_to_entities(self, text: str,
                        spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t['object'] for t in spo_list) | set(t['subject']
                                                            for t in spo_list)
        return list(entities)

    def spo_to_relations(self, text: str,
                         spo_list: List[Dict[str, str]]) -> List[str]:
        return [t['predicate'] for t in spo_list]

    def spo_to_selection(self, sfEntityMentions: str, spo_list: List[Dict[str, str]]
                         ) -> List[Dict[str, int]]:

        selection = []
        for triplet in spo_list:

            object = triplet['object']
            subject = triplet['subject']

            object_pos_begin, object_pos_end   = self.get_ner_idx(object, sfEntityMentions)
            subject_pos_begin, subject_pos_end = self.get_ner_idx(subject, sfEntityMentions)
            relation_pos = self.relation_vocab[triplet['predicate']]

            object_pos_end  += 1
            subject_pos_end += 1

            selection.append({
                'subject': subject_pos_end,
                'predicate': relation_pos,
                'object': object_pos_end
            })

        return selection

    def get_ner_idx(self, ner, sfEntityMentions):
        begin = -1
        end = -1
        for ner_str in sfEntityMentions:
            if ner == ner_str['ner_str']:
                begin = ner_str['ner_index'][0]
                end = ner_str['ner_index'][-1]
                break

        return begin, end


    def spo_to_bio(self, text: str, sfEntityMentions, entities: List[str]) -> List[str]:
        bio = ['O'] * (len(text)+1)

        for e in entities:
            begin, end = self.get_ner_idx(e, sfEntityMentions)
            begin += 1
            end   += 1

            assert end <= len(text)+1

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio

'''
if __name__=="__main__":

    hyper = Hyper(os.path.join('../../experiments',
                               'NYT_selection_re' + '.json'))
    nyt   = NYT_selection_preprocessing(hyper)
    nyt.gen_all_data()
'''

