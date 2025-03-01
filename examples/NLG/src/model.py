#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter

import loralib as lora


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

######################################## MoLEx #############################################
class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, proj_dim=256, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        self.top_k = min(num_global_experts, int(k))
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)


    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits

class GPT2MoLEx(nn.Module):
    def __init__(self, config):
        super(GPT2MoLEx, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

        self.use_gate = self.config.use_gate
        self.use_learn_weight = self.config.use_learn_weight
        self.use_indv_gate = self.config.use_indv_gate
        if self.config.use_gate and self.config.use_indv_gate:
            for layer_module in self.h:
                ################ cosine gate ##################
                if config.cosine_gate:
                    print("COSINE GATE INIT")
                    layer_module.gate = CosineTopKGate(config.n_embd, config.n_layer, proj_dim=config.proj_dim)
                ###############################################
                else:
                    layer_module.gate = nn.Linear(config.n_embd, config.n_layer)
                if self.config.use_learn_weight == 1:
                    layer_module.weight_main = nn.Parameter(torch.tensor([self.config.weight_main_init]))
                    layer_module.weight_other = nn.Parameter(torch.tensor([self.config.weight_other_init]))
                else:
                    layer_module.weight_main = self.config.weight_main_init
                    layer_module.weight_other = self.config.weight_other_init
        elif self.config.use_gate and self.config.use_indv_weight:
            for layer_module in self.h:
                layer_module.weight_main = nn.Parameter(torch.tensor([self.config.weight_main_init]))
                layer_module.weight_other = nn.Parameter(torch.tensor([self.config.weight_other_init]))
            ################ cosine gate ##################
            if config.cosine_gate:
                print("COSINE GATE INIT")
                self.gate = CosineTopKGate(config.n_embd, config.n_layer, proj_dim=config.proj_dim)
            ###############################################
            else:
                self.gate = nn.Linear(config.n_embd, config.n_layer)
        elif self.config.use_gate:
            ################ cosine gate ##################
            if config.cosine_gate:
                print("COSINE GATE INIT")
                self.gate = CosineTopKGate(config.n_embd, config.n_layer, proj_dim=config.proj_dim)
            ###############################################
            else:
                self.gate = nn.Linear(config.n_embd, config.n_layer)
            if self.config.use_learn_weight == 1:
                self.weight_main = nn.Parameter(torch.tensor([self.config.weight_main_init]))
                self.weight_other = nn.Parameter(torch.tensor([self.config.weight_other_init]))
            else:
                self.weight_main = self.config.weight_main_init
                self.weight_other = self.config.weight_other_init
        self.num_hidden_layers = config.n_layer
        self.layer_list = [0] * self.num_hidden_layers
        self.cur_layer_list = [0] * self.num_hidden_layers
        self.layers_to_use = self.config.layers_to_use
        self.cosine_gate = config.cosine_gate
        self.use_load_balance = config.use_load_balance
        self.mean_mode = config.mean_mode
        self.sig_sft = config.sig_sft
        self.use_indv_weight = config.use_indv_weight


    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []

        self.loss = 0
        for ind, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states_main, present_main = block(hidden_states, layer_past = layer_past, len_past=len_past)

            if ind < self.layers_to_use: 
                if self.use_indv_gate:
                    layer_out_all = block.gate(hidden_states)
                else:
                    layer_out_all = self.gate(hidden_states)
                
                ######### mode #########
                if not self.mean_mode:
                    if self.sig_sft:
                        layer_out = F.sigmoid(layer_out_all)
                    else:
                        # layer_out = F.softmax(layer_out_all, dim=-1)
                        layer_out = layer_out_all
                    layer_topk = torch.topk(layer_out, 1).indices
                    j = torch.mode(layer_topk.view(1,-1)).values
                ######### mean #########
                elif self.sig_sft:
                    layer_out = torch.mean(F.sigmoid(layer_out_all), dim=[0,1])
                    j = torch.topk(layer_out, 1).indices
                else:
                    layer_out = torch.mean(F.softmax(layer_out_all, dim=-1), dim=[0,1])
                    j = torch.topk(layer_out, 1).indices
                ########################
                self.layer_list[j] += 1
                self.cur_layer_list[j] += 1

                if self.use_load_balance:
                    self.loss += self.set_load_balance(layer_out_all)
                hidden_states_other, present_other = self.h[j](hidden_states, layer_past = layer_past, len_past=len_past)

                if self.use_indv_gate or self.use_indv_weight:
                    if isinstance(block.weight_main, torch.Tensor):
                        exp_weight_main = torch.exp(block.weight_main)
                        exp_weight_other = torch.exp(block.weight_other)
                        denom = exp_weight_main + exp_weight_other
                        hidden_states = (exp_weight_main/denom) * hidden_states_main + (exp_weight_other/denom) * hidden_states_other
                        present = (exp_weight_main/denom) * present_main + (exp_weight_other/denom) * present_other
                    else:
                        hidden_states = block.weight_main * hidden_states_main + block.weight_other * hidden_states_other
                        present = block.weight_main * present_main + block.weight_other * present_other
                elif self.use_gate:
                    if isinstance(self.weight_main, torch.Tensor):
                        exp_weight_main = torch.exp(self.weight_main)
                        exp_weight_other = torch.exp(self.weight_other)
                        denom = exp_weight_main + exp_weight_other
                        hidden_states = self.weight_main * hidden_states_main + self.weight_other * hidden_states_other
                        present = self.weight_main * present_main + self.weight_other * present_other
                    else:
                        hidden_states = self.weight_main * hidden_states_main + self.weight_other * hidden_states_other
                        present = self.weight_main * present_main + self.weight_other * present_other
            else:
                hidden_states = hidden_states_main
                present = present_main
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

    def set_load_balance(self, layer_out):
        score = F.softmax(layer_out, dim=-1)
        prob_expert = score.mean(dim=[0,1]) 
        fraction_expert = (torch.FloatTensor(self.cur_layer_list)/sum(self.cur_layer_list)).to(torch.device("cuda"))
        ######### OLD LOAD BALANCE #########
        loss = (fraction_expert * prob_expert).sum() * self.num_hidden_layers
        ######### NEW LOAD BALANCE #########
        # loss = (fraction_expert * prob_expert).sum()
        ####################################
        self.cur_layer_list = [0] * self.num_hidden_layers
        return loss

class GPT2MoLExFORWARD(nn.Module):
    def __init__(self, config):
        super(GPT2MoLExFORWARD, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

        self.use_gate = self.config.use_gate
        self.use_learn_weight = self.config.use_learn_weight
        self.use_indv_gate = self.config.use_indv_gate
        for ind, layer_module in enumerate(self.h):
            if config.n_layer-ind == 1:
                break
            ################ cosine gate ##################
            elif config.cosine_gate:
                print("COSINE GATE INIT")
                layer_module.gate = CosineTopKGate(config.n_embd, config.n_layer-ind, proj_dim=config.proj_dim)
            ###############################################
            else:
                layer_module.gate = nn.Linear(config.n_embd, config.n_layer-ind)
            if self.config.use_learn_weight == 1:
                layer_module.weight_main = nn.Parameter(torch.tensor([self.config.weight_main_init]))
                layer_module.weight_other = nn.Parameter(torch.tensor([self.config.weight_other_init]))
            else:
                layer_module.weight_main = self.config.weight_main_init
                layer_module.weight_other = self.config.weight_other_init

        self.num_hidden_layers = config.n_layer
        self.layer_list = [0] * self.num_hidden_layers
        self.cur_layer_list = [0] * self.num_hidden_layers
        self.layers_to_use = self.config.layers_to_use
        self.cosine_gate = config.cosine_gate
        self.use_load_balance = config.use_load_balance
        self.mean_mode = config.mean_mode
        self.sig_sft = config.sig_sft
        self.use_indv_weight = config.use_indv_weight


    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []

        self.loss = 0
        for ind, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states_main, present_main = block(hidden_states, layer_past = layer_past, len_past=len_past)

            if self.n_layer-ind-1 == 0:
                hidden_states = hidden_states_main
                present = present_main
            else:
                if self.n_layer-ind-1 > 1:
                    layer_out_all = block.gate(hidden_states)
                    
                    ######### mode #########
                    if not self.mean_mode:
                        if self.sig_sft:
                            layer_out = F.sigmoid(layer_out_all)
                        else:
                            # layer_out = F.softmax(layer_out_all, dim=-1)
                            layer_out = layer_out_all
                        layer_topk = torch.topk(layer_out, 1).indices
                        j = torch.mode(layer_topk.view(1,-1)).values
                    ######### mean #########
                    elif self.sig_sft:
                        layer_out = torch.mean(F.sigmoid(layer_out_all), dim=[0,1])
                        j = torch.topk(layer_out, 1).indices
                    else:
                        layer_out = torch.mean(F.softmax(layer_out_all, dim=-1), dim=[0,1])
                        j = torch.topk(layer_out, 1).indices
                        ########################
                    self.layer_list[j] += 1
                    self.cur_layer_list[j] += 1
                    if self.use_load_balance:
                        self.loss += self.set_load_balance(layer_out_all, ind)
                elif self.n_layer-ind-1 == 1:
                    j = ind+1

                hidden_states_other, present_other = self.h[j](hidden_states, layer_past = layer_past, len_past=len_past)

                if isinstance(block.weight_main, torch.Tensor):
                    exp_weight_main = torch.exp(block.weight_main)
                    exp_weight_other = torch.exp(block.weight_other)
                    denom = exp_weight_main + exp_weight_other
                    hidden_states = (exp_weight_main/denom) * hidden_states_main + (exp_weight_other/denom) * hidden_states_other
                    present = (exp_weight_main/denom) * present_main + (exp_weight_other/denom) * present_other
                else:
                    hidden_states = block.weight_main * hidden_states_main + block.weight_other * hidden_states_other
                    present = block.weight_main * present_main + block.weight_other * present_other

            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

    def set_load_balance(self, layer_out, ind):
        score = F.softmax(layer_out, dim=-1)
        prob_expert = score.mean(dim=[0,1]) 
        fraction_expert = (torch.FloatTensor(self.cur_layer_list)/sum(self.cur_layer_list)).to(torch.device("cuda"))
        ######### OLD LOAD BALANCE #########
        loss = (fraction_expert[ind:] * prob_expert).sum() * self.num_hidden_layers
        ######### NEW LOAD BALANCE #########
        # loss = (fraction_expert * prob_expert).sum()
        ####################################
        self.cur_layer_list = [0] * self.num_hidden_layers
        return loss
    
##############################################################################################

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config


    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
        use_gate=False,
        use_learn_weight=1,
        use_indv_gate=False,
        cosine_gate=False,
        proj_dim=256,
        weight_main_init=0.95,
        weight_other_init=0.05,
        layers_to_use=12,
        use_load_balance=False,
        g_balance=0.01,
        mean_mode=False,
        sig_sft=False,
        use_indv_weight=False,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout

        ############### MoLEx #################
        self.use_gate = use_gate
        self.use_learn_weight = use_learn_weight
        self.use_indv_gate = use_indv_gate
        self.cosine_gate = cosine_gate
        self.proj_dim = proj_dim
        self.weight_main_init = weight_main_init
        self.weight_other_init = weight_other_init
        self.layers_to_use = layers_to_use
        self.use_load_balance = use_load_balance
        self.g_balance = g_balance
        self.mean_mode = mean_mode
        self.sig_sft = sig_sft
        self.use_indv_weight = use_indv_weight
        #######################################

class GPT2LMModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        ############### MoLEx #################
        if config.use_gate:
            self.transformer = GPT2MoLEx(config)
        elif config.layers_to_use==-2:
            self.transformer = GPT2MoLExFORWARD(config)
        else:
            self.transformer = GPT2Model(config)
        self.g_balance = config.g_balance
        #######################################
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(
        self, 
        input_ids, 
        lm_labels=None, 
        lm_mask=None, 
        past=None, 
        len_past=None, 
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                
                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break  

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

                #_t1_acc = _t1_acc * 1.0 / _batch
                #_all_acc = _all_acc * 1.0 / _batch

            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask 

            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            ########## MoLEx #########
            if hasattr(self.transformer, "use_load_balance"):
                if is_report_accuracy and self.transformer.use_load_balance:
                    return lm_logits, loss, self.transformer.loss, _t1_acc, _all_acc
                elif self.transformer.use_load_balance:
                    return lm_logits, loss, self.transformer.loss
                else:
                    return lm_logits, loss, 0
            elif is_report_accuracy:
                return lm_logits, loss, 0, _t1_acc, _all_acc
            else:
                return lm_logits, loss, 0
            ##########################
        return lm_logits, presents
           
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied()
