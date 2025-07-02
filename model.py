from logging import BASIC_FORMAT
from numpy.core.numeric import base_repr
import torch
import torch.nn as nn
import texar.torch as tx
from texar.torch.utils.utils import sequence_mask
from texar.torch.data import embedding
import numpy as np
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算p和q的两两kl散度，最终输出为batch*batch的矩阵
def my_kl_mat(p,q):
    p_mu = p[:,0]
    p_sigmasquare = p[:,1]
    q_mu = q[:,0]
    q_sigmasquare = q[:,1]
    batchsize = p_mu.shape[0]
    gaussdim = p_mu.shape[1]
    p_mu = p_mu.expand(batchsize,batchsize,gaussdim).transpose(0,1)
    p_sigmasquare = p_sigmasquare.expand(batchsize,batchsize,gaussdim).transpose(0,1)
    q_mu = q_mu.expand(batchsize,batchsize,gaussdim)
    q_sigmasquare = q_sigmasquare.expand(batchsize,batchsize,gaussdim)
    p_scale = torch.sqrt(p_sigmasquare) #标准差
    p_loc = p_mu #均值
    q_scale = torch.sqrt(q_sigmasquare) #标准差
    q_loc = q_mu #均值
    var_ratio = (p_scale / q_scale).pow(2)
    t1 = ((p_loc - q_loc) / q_scale).pow(2)
    p_q_kl = 0.5*(var_ratio + t1 - 1 - var_ratio.log())
    p_q_kl = p_q_kl.sum(dim=-1)
    return p_q_kl

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, pos_score, neg_score):
        '''
        pos_score: (batch)
        neg_score: (batch)
        '''

        return torch.mean(self.relu(neg_score - pos_score + self.margin))
    

class MORE_CL(nn.Module):
    def __init__(self, config_model, config_data):
        super(MORE_CL, self).__init__()
        self.config_model = config_model
        self.moco_m = config_model.moco_m
        self.vocab_size = config_model.encoder['vocab_size']

        self.encoder_q = EventEncoder(config_model, config_data)
        self.memory_bank = Memory_Bank(self.config_model.bank_size,dim=self.config_model.hidden_dim)
        
        self.mask_fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, self.vocab_size))
        
        # # 预测高斯分布的均值
        # self.gauss_mean_fc = nn.Sequential(
        #     nn.Linear(config_model.hidden_dim, self.config_model.gauss_dim))
        
        # # 预测高斯分布的方差
        # self.gauss_var_fc = nn.Sequential(
        #     nn.Linear(config_model.hidden_dim, self.config_model.gauss_dim))
    
    def forward(self, batch):
        evt_k, evt_k_lengths = batch.evt_k_ids, batch.evt_k_lengths
        q, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)  # queries: NxC
        k1, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        # k2, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        k2 = ""
        evt_r, evt_r_lengths = batch.evt_r_ids, batch.evt_r_lengths
        r, q_outputs = self.encoder_q(evt_r, evt_r_lengths, is_train=True)
        # r = ""

        evt_pair, evt_pair_lengths = batch.evt_pair_ids, batch.evt_pair_lengths
        pair, pair_outputs = self.encoder_q(evt_pair, evt_pair_lengths, is_train=True)
        
        # r_list = []
        # for i in range(9):
        #     evt_r, evt_r_lengths = batch.evt_r_ids_list[i], batch.evt_r_lengths_list[i]
        #     r, r_outputs = self.encoder_q(evt_r, evt_r_lengths, is_train=True)
        #     r_list.append(r)
        
        # MLM任务
        evt_q, evt_q_lengths = batch.evt_q_ids, batch.evt_q_lengths
        mask_pos, mask_id = batch.mask_pos, batch.mask_id
        m, m_outputs = self.encoder_q(evt_q, evt_q_lengths, is_train=True)
        mlm_logits = [ex[mask_pos[idx],:].unsqueeze(0) for idx,ex in enumerate(m_outputs)]
        mlm_logits = torch.cat(mlm_logits,dim=0)
        mlm_logits = self.mask_fc(mlm_logits)

        return q, k1, k2, r, mlm_logits, pair

    # def forward(self, batch):
    #     evt_q, evt_q_lengths = batch.evt_q_ids, batch.evt_q_lengths
    #     evt_k, evt_k_lengths = batch.evt_k_ids, batch.evt_k_lengths
    #     evt_p, evt_p_lengths = batch.evt_p_ids, batch.evt_p_lengths
    #     mask_pos, mask_id = batch.mask_pos, batch.mask_id
    #     q, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)  # queries: NxC
    #     # q = nn.functional.normalize(q, dim=1)
    #     # q_mean = self.gauss_mean_fc(q)
    #     # q_var = torch.exp(self.gauss_var_fc(q))
    #     # q_mean = nn.functional.normalize(q_mean, dim=1).unsqueeze(1)
    #     # q_var = nn.functional.normalize(q_var, dim=1).unsqueeze(1)
    #     # q = torch.cat((q_mean,q_var),dim=1)
        

    #     k1, k1_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
    #     # k1 = nn.functional.normalize(k1, dim=1)
    #     # k1_mean = self.gauss_mean_fc(k1)
    #     # k1_var = torch.exp(self.gauss_var_fc(k1))
    #     # k1_mean = nn.functional.normalize(k1_mean, dim=1).unsqueeze(1)
    #     # k1_var = nn.functional.normalize(k1_var, dim=1).unsqueeze(1)
    #     # k1 = torch.cat((k1_mean,k1_var),dim=1)
        
        
    #     k2, k2_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
    #     # k2 = nn.functional.normalize(k2, dim=1)
    #     # k2_mean = self.gauss_mean_fc(k2)
    #     # k2_var = torch.exp(self.gauss_var_fc(k2))
    #     # k2_mean = nn.functional.normalize(k2_mean, dim=1).unsqueeze(1)
    #     # k2_var = nn.functional.normalize(k2_var, dim=1).unsqueeze(1)
    #     # k2 = torch.cat((k2_mean,k2_var),dim=1)
        
        
    #     p, p_outputs = self.encoder_q(evt_p, evt_p_lengths, is_train=True)
    #     # p = nn.functional.normalize(p, dim=1)
    #     # p_mean = self.gauss_mean_fc(p)
    #     # p_var = torch.exp(self.gauss_var_fc(p))
    #     # p_mean = nn.functional.normalize(p_mean, dim=1).unsqueeze(1)
    #     # p_var = nn.functional.normalize(p_var, dim=1).unsqueeze(1)
    #     # p = torch.cat((p_mean,p_var),dim=1)
        
        
    #     m, m_outputs = self.encoder_q(evt_q, evt_q_lengths, is_train=True)
    #     mlm_logits = [ex[mask_pos[idx],:].unsqueeze(0) for idx,ex in enumerate(m_outputs)]
    #     mlm_logits = torch.cat(mlm_logits,dim=0)
    #     mlm_logits = self.mask_fc(mlm_logits)
    #     # return q_mean, q_var, k1_mean, k1_var, k2_mean, k2_var, p_mean, p_var, mlm_logits
    #     return q, k1, k2, p, mlm_logits


        

class EventEncoder(nn.Module):
    def __init__(self,config_model, config_data):
        super(EventEncoder, self).__init__()
        self.config_model = config_model
        self.config_data = config_data
        self.input_fc = nn.Linear(config_model.word_dim, config_model.hidden_dim)
        self.encoder = tx.modules.BERTEncoder(
            hparams=self.config_model.encoder)

        self.fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, config_model.hidden_dim))
        
        # 预测高斯分布的均值
        self.gauss_mean_fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, config_model.gauss_dim))
        
        # 预测高斯分布的方差
        self.gauss_var_fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, config_model.gauss_dim),)


    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        
        word_embed = self.word_embedder(tokens)
        pos_embed = self.pos_embedder(positions)

        return word_embed + pos_embed
    def forward(self, event_ids, event_lengths, is_train=False):
        encoder_input, event_lengths = event_ids, event_lengths
        batch_size = len(encoder_input)
        
        encoder_input_length = event_lengths
        # positions = torch.arange(
        #     encoder_input_length.max(), dtype=torch.long,
        #     device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # enc_input_embedding = self.input_fc(self._embedding_fn(encoder_input, positions))
        # enc_input_embedding = self._embedding_fn(encoder_input, positions)

        outputs, pooled_output = self.encoder(
            inputs=encoder_input, sequence_length=encoder_input_length)
        inputs_padding = sequence_mask(
            event_lengths, event_ids.size()[1]).float()
        
        # event_embedding = (outputs * inputs_padding.unsqueeze(-1))[:,1:].sum(1)/(event_lengths.unsqueeze(-1)-1)
        # event_embedding = self.fc(enc_output[:,0,:])
        event_embedding = outputs[:,0,:]
        # event_embedding = self.fc(event_embedding)

        event_embedding = nn.functional.normalize(event_embedding, dim=1)
        event_embedding_mean = self.gauss_mean_fc(event_embedding).unsqueeze(1)
        event_embedding_var = torch.exp(self.gauss_var_fc(event_embedding)).unsqueeze(1)
        # event_embedding_mean = nn.functional.normalize(event_embedding_mean, dim=1).unsqueeze(1)
        # event_embedding_var = nn.functional.normalize(event_embedding_var, dim=1).unsqueeze(1)
        # event_embedding = torch.cat((event_embedding_mean,event_embedding_var),dim=1)

        if is_train:
            return event_embedding, outputs
        else:
            # return event_embedding
            return event_embedding,event_embedding_mean,event_embedding_var

class Func_Relation_Attention(nn.Module):
    def __init__(self, gauss_dim):
        super(Func_Relation_Attention, self).__init__()
        self.w = nn.Parameter(torch.randn(10,2,gauss_dim)) # [10,2,500]
    def forward(self, q, r=None, pair=None, i=None, attention=False,):
        """
        q [125,2,500]
        r [125,2,500]
        pair [125,2,500]
        i int [0-9]
        """
        if attention:
            alpha = (pair.unsqueeze(1) * self.w).sum(-1).softmax(dim=1)  # [125,10,2]
            w_e = (alpha.unsqueeze(-1) * self.w).sum(dim=1)  # [125,2,500]
            w_e = nn.functional.normalize(w_e,dim=-1) # [125,2,500]
            q_res = q - (q * w_e).sum(-1).unsqueeze(-1) * w_e
            r_res = r - (r * w_e).sum(-1).unsqueeze(-1) * w_e
            return q_res, r_res
        else:
            w = self.w[i]
            w = nn.functional.normalize(w, dim=-1)
            q_res = q - (q * w).sum(-1).unsqueeze(-1) * w
            return q_res

class Func_Relation(nn.Module):
    def __init__(self, gauss_dim):
        super(Func_Relation, self).__init__()
        self.w = nn.Parameter(torch.randn(2,gauss_dim))
    def forward(self, q):
        w = nn.functional.normalize(self.w, dim=-1)
        q_res = q - (q * w).sum(-1).unsqueeze(-1) * w
        return q_res

class Func_Relation_List(nn.Module):
    def __init__(self, gauss_dim):
        super(Func_Relation_List, self).__init__()
        self.Func_Relation_list = nn.ParameterList()
        for i in range(10):
            f = Func_Relation(gauss_dim)
            self.Func_Relation_list.append(f)
    def forward(self, q):
        pass

# class Func_Relation_mat(nn.Module):
#     def __init__(self, gauss_dim):
#         super(Func_Relation, self).__init__()
#         self.w = nn.Parameter(torch.randn(9,2,gauss_dim))
#     def forward(self, q):
#         w = nn.functional.normalize(self.w, dim=-1).unsqueeze(1)
#         q_res = q - (q * w).sum(-1).unsqueeze(-1) * w
#         return q_res

class Memory_Bank(nn.Module):
    def __init__(self, bank_size, dim):
        super(Memory_Bank, self).__init__()
        # self.W = nn.Parameter(torch.randn(dim, bank_size))
        self.W_mean = nn.Parameter(torch.randn(dim, bank_size))
        self.W_var = nn.Parameter(torch.randn(dim, bank_size))
        self.sigmoid = nn.Sigmoid()
    def forward(self, q):
        # memory_bank = self.W
        # memory_bank = nn.functional.normalize(memory_bank, dim=0)

        memory_bank_mean = self.W_mean
        memory_bank_mean = nn.functional.normalize(memory_bank_mean, dim=0).t().unsqueeze(1)

        memory_bank_var = self.W_var
        memory_bank_var = nn.functional.normalize(self.sigmoid(memory_bank_var), dim=0).t().unsqueeze(1)

        memory_bank = torch.cat((memory_bank_mean,memory_bank_var),dim=1)

        logit = -(my_kl_mat(q,memory_bank) + my_kl_mat(memory_bank,q).t())/2

        # logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return logit

class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self,  # type: ignore
                output: torch.Tensor,
                target: torch.Tensor,
                label_lengths: torch.LongTensor) -> torch.Tensor:
        r"""Compute the label smoothing loss.
        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])

        return tx.losses.sequence_softmax_cross_entropy(
            labels=model_prob,
            logits=output,
            sequence_length=label_lengths,
            average_across_batch=False,
            sum_over_timesteps=False,
        )