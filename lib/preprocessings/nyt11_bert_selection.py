import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property

from lib.config import Hyper
from pytorch_pretrained_bert import BertTokenizer

import nltk
#nltk.download('punkt')

class NYT11_bert_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

        '''

        self.relation_vocab_path = os.path.join('../../data/nyt11/multi_head_selection/',
                                                hyper.relation_vocab)
        '''

        vocab_path = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)


    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r'))

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        return None

    def gen_vocab(self, min_freq: int):
        return None

    def gen_ERNIE_vocab(self):
        return None

    def _clean_whitespace(self, text):
        text_cut = nltk.word_tokenize(text)
        text_x = []
        for word in text_cut:
            text_x.append(word)
            text_x.append(' ')

        text_x = ''.join(text_x)
        return text_x.strip()

    def _add__clean_whitespace(self, text):
        text_x = []
        for word in text:
            text_x.append(word)
            text_x.append(' ')

        text_x = ''.join(text_x)
        return text_x.strip()

    def wordpiece2str(self, wordpiece):
        text_cut = []
        for word in wordpiece:
            text_cut.append(word)
            text_cut.append(' ')
        text_cut = ''.join(text_cut)

        return text_cut.strip()

    def text_to_bio(self, bio, begin, end):

        bio[begin] = 'B'
        for i in range(begin + 1, end + 1):
            bio[i] = 'I'

        return bio

    def find_ner_pos(self, text, ner):

        pos = text.index(ner[0])
        if len(ner) > 1:
            pos_next = text.index(ner[1])
            if pos+1 == pos_next:
                return pos

        return pos

    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text_org        = instance['sentext']
        #sfEntityMentions = instance['sfEntityMentions']

        bio = None
        selection = []
        ner_tokenizer = []
        spo_list  = []

        if 'relations' in instance:
            triples_list = instance['relations']

            text = self._clean_whitespace(text_org)
            tokenizer_text = self.tokenizer.tokenize(text)
            #text = self._add__clean_whitespace(tokenizer_text)
            if len(tokenizer_text) > self.hyper.max_text_len - 1:
                return None

            if not self._check_valid(text_org, triples_list):
                return None

            for spo in triples_list:
                subject_org_ner = spo['em1']
                object_org_ner  = spo['em2']

                subject_tokenizer_ner = self.tokenizer.tokenize(subject_org_ner)
                object_tokenizer_ner  = self.tokenizer.tokenize(object_org_ner)
                subject_ner = self.wordpiece2str(subject_tokenizer_ner)
                object_ner  = self.wordpiece2str(object_tokenizer_ner)

                subject_ner_pos = self.find_ner_pos(tokenizer_text, subject_tokenizer_ner)
                object_ner_pos  = self.find_ner_pos(tokenizer_text, object_tokenizer_ner)

                subject_begin = subject_ner_pos
                subject_end   = subject_ner_pos + len(subject_tokenizer_ner) - 1

                object_begin = object_ner_pos
                object_end   = object_ner_pos + len(object_tokenizer_ner) - 1

                bio = ['O']*(len(tokenizer_text)+1)
                subject_begin += 1
                subject_end   += 1

                object_begin  += 1
                object_end    += 1

                bio = self.text_to_bio(bio, subject_begin, subject_end)
                bio = self.text_to_bio(bio, object_begin, object_end)

                relation_pos = self.relation_vocab[spo['rtext']]

                '''
                ner_tokenizer.append({relation_pos: 'predicate',
                                      object_ner: object_org_ner,
                                      subject_ner: subject_org_ner
                                      })
                '''
                ner_tokenizer.append({'predicate': relation_pos,
                                      object_org_ner: object_ner,
                                      subject_org_ner: subject_ner
                                      })
                selection.append({
                    'subject': subject_end,
                    'predicate': relation_pos,
                    'object': object_end
                })

                spo_list.append({
                    'predicate': spo['rtext'],
                    'object': spo['em2'],
                    'subject': spo['em1']
                 })

        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection,
            'ner_tokenizer': ner_tokenizer
        }
        return json.dumps(result, ensure_ascii=False)

    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        #source  = '../../raw_data/nyt11/train.json'
        #target  = '../../data/nyt11/multi_head_selection/train.json'
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
            if t['em2'] not in text or t['em1'] not in text:
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

            object_pos_end += 1
            subject_pos_end+= 1

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
                               'NYT11_bert_selection_re' + '.json'))
    nyt   = NYT_bert_selection_preprocessing(hyper)
    nyt.gen_all_data()
'''
