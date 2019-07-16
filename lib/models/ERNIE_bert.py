import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

class BERT(nn.Module):
    def __init__(self, pre_bert_path):
        super(BERT, self).__init__()

        self.bert   = BertModel.from_pretrained(pre_bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, seqs, attention_mask=None, output_all_encoded_layers=False):
        encoder_seq, pooled_output = self.bert(seqs, attention_mask=attention_mask, output_all_encoded_layers=output_all_encoded_layers)

        return encoder_seq, pooled_output

class bert_embedding(nn.Module):
    def __init__(self, pre_bert_path):
        super(bert_embedding, self).__init__()
        #device_id = [0, 1, 2]
        self.bert  = BERT(pre_bert_path)
        self.bert  = nn.DataParallel(self.bert)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens, mask=None):
        seqs, K = self.bert(tokens.long(), attention_mask=mask)
        seqs  = self.dropout(seqs)

        return seqs, K

class bertBaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super(bertBaseModel, self).__init__(config)

        self.bert_model  = BertModel(config)
        self.dropout     = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        encoder_output, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, \
                                                    output_all_encoded_layers=False)
        pooled_output                 = self.dropout(pooled_output)
        encoder_output                = self.dropout(encoder_output)

        return encoder_output, pooled_output

class self_bert_embedding(nn.Module):
    def __init__(self, vocab_size):
        super(self_bert_embedding, self).__init__()
        device_id = [0,1,2]
        self.bert_config = BertConfig(vocab_size)
        self.models  = bertBaseModel(self.bert_config)
        self.models = nn.DataParallel(self.models, device_id)

    def forward(self, input_ids, mask=None):
        encoder_output, pooled_output = self.models(input_ids, attention_mask=mask)

        return encoder_output, pooled_output
