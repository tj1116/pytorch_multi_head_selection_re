from typing import Dict, List, Tuple, Set, Optional


class F1_triplet(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

        self.tag_A = 1e-10
        self.tag_B = 1e-10
        self.tag_C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

        self.tag_A = 1e-10
        self.tag_B = 1e-10
        self.tag_C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B +
                                 self.C), self.A / self.B, self.A / self.C

        '''
        tag_f1, tag_p, tag_r = 2 * self.tag_A / (self.tag_B +
                                 self.tag_C), self.tag_A / self.tag_B, self.tag_A / self.tag_C

        
        result = {"precision": p, "recall": r, "fscore": f1, \
                  "tag_precision": tag_p, "tag_recall": tag_r, "tag_fscore": tag_f1}
        '''
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    '''
    def __call__(self, predictions: List[List[Dict[str, str]]],
                 gold_labels: List[List[Dict[str, str]]]):
    '''
    def __call__(self, predictions, gold_labels):

        for g, p in zip(gold_labels['selection_triplets'], predictions['spo_gold']):
            g_set = set('_'.join((gg['object'], gg['predicate'],
                                  gg['subject'])) for gg in g)
            p_set = set('_'.join((pp['object'], pp['predicate'],
                                  pp['subject'])) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)

        '''
        for g, p in zip(gold_labels['tag_gold'], predictions['tag_pred']):
            for i in range(len(g)):
                if g[i] != 2 and g[i] != 5:
                    self.tag_C += 1

                if p[i] != 2 and g[i] != 5:
                    self.tag_B += 1
                    if p[i] == g[i]:
                        self.tag_A += 1
        '''
            