import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

#from torchcrf import CRF

from lib.models.crf import CRF
from lib.models.ERNIE_bert import bert_embedding, self_bert_embedding


class MultiHeadSelection(nn.Module):
    def __init__(self, hyper, is_rnn=True) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.is_rnn = is_rnn

        if self.hyper.is_bert == 'ERNIE':
            self.word_vocab = json.load(
                open(os.path.join(self.data_root, 'word_ERNIE_vocab.json'), 'r'))

        elif self.hyper.is_bert == "bert_bilstem_crf" or \
                self.hyper.is_bert == "nyt_bert_tokenizer"  or \
                self.hyper.is_bert == "nyt11_bert_tokenizer" or \
                self.hyper.is_bert == "nyt10_bert_tokenizer":
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


        if self.hyper.is_bert == 'ERNIE' or \
                self.hyper.is_bert == "bert_bilstem_crf" or \
                self.hyper.is_bert == "nyt_bert_tokenizer" or \
                self.hyper.is_bert == "nyt11_bert_tokenizer" or \
                self.hyper.is_bert == "nyt10_bert_tokenizer":
            self.word_embeddings = bert_embedding(self.hyper.bert_pretrain_path)

        elif self.hyper.is_bert == 'bert_crf':
            self.word_embeddings = self_bert_embedding(len(self.word_vocab))

        else:
            self.word_embeddings = nn.Embedding(num_embeddings=len(
                self.word_vocab),
                embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.rel_emb_size)

        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        else:
            raise ValueError('cell name should be gru/lstm!')

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)
        #self.tagger = CRF(len(self.bio_vocab), batch_first=True)

        self.selection_u = nn.Linear(hyper.hidden_size + hyper.rel_emb_size,
                                     hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.rel_emb_size,
                                     hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        # remove <pad>
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)
        #self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab))

        # self.accuracy = F1Selection()

    def inference(self, mask, text_list, decoded_tag, selection_logits, ner_tokenizer=None):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
            -1, -1, len(self.relation_vocab),
            -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold

        selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   selection_tags, ner_tokenizer)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
            -1, -1, len(self.relation_vocab),
            -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.cuda(self.gpu)
        selection_gold = sample.selection_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)
        text_list = sample.text
        spo_gold = sample.spo_gold
        ner_tokenizer = None

        if not is_train and self.hyper.is_bert != 'ERNIE' and self.hyper.is_bert != "bert_bilstem_crf":
            ner_tokenizer = sample.ner_tokenizer

        if self.hyper.is_bert == 'ERNIE' or \
                self.hyper.is_bert == "bert_bilstem_crf" or \
                self.hyper.is_bert == "nyt_bert_tokenizer" or \
                self.hyper.is_bert == "nyt11_bert_tokenizer" or \
                self.hyper.is_bert == "nyt10_bert_tokenizer":
            mask = tokens != 0
            bert_embedded, _= self.word_embeddings(tokens, mask=mask)

            if self.is_rnn:
                o, h = self.encoder(bert_embedded)

        else:
            mask = tokens != 0 #self.word_vocab['<pad>']  # batch x seq
            embedded = self.word_embeddings(tokens)
            o, h = self.encoder(embedded)

        if self.is_rnn:
            o = (lambda a: sum(a) / 2)(torch.split(o,
                                               self.hyper.hidden_size,
                                               dim=2))

            emi = self.emission(o)
        else:
            emi = self.emission(bert_embedded)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = self.tagger(emi, bio_gold, mask=mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=mask)
            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)


        tag_emb = self.bio_emb(bio_gold)

        if self.is_rnn:
            o = torch.cat((o, tag_emb), dim=2)
        else:
            o = torch.cat((bert_embedded, tag_emb), dim=2)

        # forward multi head selection
        u = self.activation(self.selection_u(o)).unsqueeze(1)
        v = self.activation(self.selection_v(o)).unsqueeze(2)
        u = u + torch.zeros_like(v)
        v = v + torch.zeros_like(u)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        if not is_train:
            output['selection_triplets'] = self.inference(
                mask, text_list, decoded_tag, selection_logits, ner_tokenizer)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits,
                                                 selection_gold)
        #selection_loss = 150*selection_loss
        #crf_loss = 150*crf_loss
        #crf_loss = 10 * crf_loss
        loss = crf_loss + selection_loss

        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)
        return output

    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor,
                         ner_tokenizer = None
                         ) -> List[List[Dict[str, str]]]:

        reversed_relation_vocab = {
            v: k for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []
            entity_str = ''

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))

                if self.hyper.is_bert != 'ERNIE':
                    for i in range(len(entity)):
                        if i == 0:
                            entity_str = entity[i]
                        else:
                            entity_str = entity_str + ' ' + entity[i]
                    return entity_str

            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'None':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)

            if self.hyper.is_bert != "bert_bilstem_crf":
                label_ner = ner_tokenizer[b]
                for ner in label_ner:
                    '''
                    ner_list = {j:i for i,j in ner.items()}
                    if p in ner_list and object in ner_list and subject in ner_list:
                        object  = ner_list[object]
                        subject = ner_list[subject]
                        break
                    '''
                    if p in ner and object in ner and subject in ner:
                        object  = ner[object]
                        subject = ner[subject]
                        break


            #assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass

