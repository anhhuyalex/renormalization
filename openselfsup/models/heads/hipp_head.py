import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

#from ..hopfield_modules import Hopfield
from ..registry import HEADS
from .. import builder
from .rnn_modules import SelfLSTM, DWSelfLSTM
from . import rnn_modules
from . import aha_modules

@HEADS.register_module
class HippHead(nn.Module):
    def __init__(self, hipp_mlp, pred_mlp):
        super().__init__()
        self.hipp_mlp = builder.build_neck(hipp_mlp)
        self.pred_mlp = builder.build_neck(pred_mlp)

    def init_weights(self, init_linear='kaiming'):
        self.hipp_mlp.init_weights(init_linear=init_linear)
        self.pred_mlp.init_weights(init_linear=init_linear)

    def get_hipp_loss(self, input_embd):
        two_time_embd = torch.cat(
                [input_embd[:-1].unsqueeze(1), input_embd[1:].unsqueeze(1)],
                dim=1)
        two_time_embd = two_time_embd.reshape(two_time_embd.size(0), -1)
        hipp_resp = self.hipp_mlp([two_time_embd])[0]
        hipp_resp = nn.functional.normalize(hipp_resp, dim=1)
        pred_input = torch.cat([hipp_resp, input_embd[1:]], dim=1)
        pred_output = self.pred_mlp([pred_input])[0]
        pred_output = pred_output[:-1]
        pred_output = nn.functional.normalize(pred_output, dim=1)
        pred_loss = 2 - 2 * (pred_output * input_embd[2:]).sum() \
                          / pred_output.size(0)
        loss = dict(
                loss=pred_loss, 
                pred_loss=pred_loss)
        return loss

    def forward(self, input_embd, two_views=True):
        if two_views:
            input_embd = input_embd.reshape(-1, 2, input_embd.size(1))
            loss_view0 = self.get_hipp_loss(input_embd[:, 0, :])
            loss_view1 = self.get_hipp_loss(input_embd[:, 1, :])
            final_loss = {}
            for loss_key in loss_view0:
                final_loss[loss_key] = loss_view0[loss_key] + loss_view1[loss_key]
        else:
            final_loss = self.get_hipp_loss(input_embd)
        return final_loss


@HEADS.register_module
class HippRNNHead(nn.Module):
    def __init__(
            self, pred_mlp, rnn_kwargs,
            rnn_type='default', rnn_tile=None,
            pass_states=None):
        super().__init__()
        self.rnn_tile = rnn_tile
        
        self.pred_mlp = builder.build_neck(pred_mlp)
        self.rnn_type = rnn_type
        self.rnn_kwargs = rnn_kwargs
        self.pass_states = pass_states
        self.last_states = None
        self.build_rnn()

    def build_multi_rnns(self, num_rnns):
        assert isinstance(self.rnn_kwargs, list) \
                or isinstance(self.rnn_kwargs, tuple)
        assert len(self.rnn_kwargs) == num_rnns
        self.rnn = []
        for idx in range(num_rnns):
            self.rnn.append(Hopfield(**self.rnn_kwargs[idx]))

    def build_rnn(self):
        if self.rnn_type == 'default':
            self.rnn = nn.LSTM(**self.rnn_kwargs)
        elif self.rnn_type == 'relu':
            self.rnn = SelfLSTM(activation='relu', **self.rnn_kwargs)
        elif self.rnn_type == 'self':
            self.rnn = SelfLSTM(**self.rnn_kwargs)
        elif self.rnn_type == 'simple_rnn':
            self.rnn = rnn_modules.SelfRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dw_simple_rnn':
            self.rnn = rnn_modules.DWSelfRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dwt_simple_rnn':
            self.rnn = rnn_modules.DWTransSelfRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dw':
            self.rnn = DWSelfLSTM(**self.rnn_kwargs)
        elif self.rnn_type == 'dwt':
            self.rnn = rnn_modules.DWTransSelfLSTM(**self.rnn_kwargs)
        elif self.rnn_type == 'pat':
            self.rnn = rnn_modules.NaivePatSoftmaxRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'gnrl_pat':
            self.rnn = rnn_modules.GnrlPatSoftmaxRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'sim_gnrl_pat':
            self.rnn = rnn_modules.SimGnrlPatSoftmaxRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'mltmx_gnrl_pat':
            self.rnn = rnn_modules.MultiSoftmaxGnrlPatRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'gate_sep_hp':
            self.rnn = rnn_modules.GateSepHPathRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'gate_sep_hp_fw':
            self.rnn = rnn_modules.FWGateSepHPathRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'fxec_gate_sep_hp':
            self.rnn = rnn_modules.FxEcGateSepHPathRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'leg_mask_fxec_gate_sep_hp':
            self.rnn = rnn_modules.LegMaskFxEcGSepHPRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dg_ca3_gate_sep_hp':
            self.rnn = rnn_modules.DGCA3GateSepHPathRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dg_dynca3_gate_sep_hp':
            self.rnn = rnn_modules.DGDynCA3GSepHPRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'dg_dynca3ca1_gate_sep_hp':
            self.rnn = rnn_modules.DGDynCA3CA1GSepHPRNN(**self.rnn_kwargs)
        elif self.rnn_type == 'hopfield':
            self.rnn = Hopfield(**self.rnn_kwargs)
        elif self.rnn_type == 'fastweights':
            self.rnn = rnn_modules.FastWeights(**self.rnn_kwargs)
        elif self.rnn_type == 'stm_aha':
            self.rnn = aha_modules.STM_AHA(**self.rnn_kwargs)
        elif self.rnn_type == 'none':
            self.rnn = nn.Identity()
        elif self.rnn_type.startswith('stacked_hopfield_'):
            num_rnns = int(self.rnn_type.replace('stacked_hopfield_', ''))
            self.build_multi_rnns(num_rnns)
            self.rnn = nn.Sequential(*self.rnn)
        elif self.rnn_type.startswith('parallel_hopfield_'):
            num_rnns = int(self.rnn_type.replace('parallel_hopfield_', ''))
            self.build_multi_rnns(num_rnns)
            self.rnn = nn.ModuleList(self.rnn)
        else:
            raise NotImplementedError

    def init_weights(self, init_linear='kaiming'):
        self.pred_mlp.init_weights(init_linear=init_linear)

    def repeat_inputs(self, input_embd):
        input_embd = torch.cat(
                [input_embd[:-1].unsqueeze(1)\
                                .repeat(1, self.rnn_tile, 1, 1)\
                                .view(-1, input_embd.size(1), 
                                      input_embd.size(2)),
                 input_embd[-1:]],
                dim=0)
        #input_embd = torch.cat(
        #        [input_embd[:-1].repeat(self.rnn_tile, 1, 1), input_embd[-1:]],
        #        dim=0)
        return input_embd

    def get_leg_mask(self, target):
        return self.rnn.get_leg_mask(target)

    def get_additional_loss(self):
        return self.rnn.get_additional_loss()

    def forward_pred_all(self, input_embd):
        # input_embd shape: [seq_len, batch_size, input_size]
        if self.rnn_tile is not None:
            input_embd = self.repeat_inputs(input_embd)

        if self.rnn_type in ['gate_sep_hp']:
            if self.pass_states is not None:
                if self.pass_states.startswith('noisy_pass:') \
                        and (self.last_states is not None):
                    noise_norm = float(self.pass_states.split(':')[1])
                    noise = torch.randn(*self.last_states.shape).cuda() \
                            * noise_norm
                    self.last_states += noise
            rnn_output, all_states = self.rnn.forward_pred_all(
                    input_embd, self.last_states)
            if self.pass_states is not None:
                if self.pass_states == 'pass_last' \
                        or self.pass_states.startswith('noisy_pass:'):
                    self.last_states = all_states[-1]
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        all_pred_outputs = []
        for each_rnn_output in rnn_output:
            _pred_output = self.pred_mlp([each_rnn_output])[0]
            all_pred_outputs.append(_pred_output)
        all_pred_outputs = torch.stack(all_pred_outputs)
        return all_pred_outputs

    def forward(self, input_embd):
        # input_embd shape: [seq_len, batch_size, input_size]
        if self.rnn_tile is not None:
            input_embd = self.repeat_inputs(input_embd)

        if self.rnn_type in ['default', 'relu', 'self', 
                             'dw', 'simple_rnn', 'dw_simple_rnn', 
                             'dwt', 'dwt_simple_rnn', 'pat', 'gnrl_pat',
                             'sim_gnrl_pat', 'mltmx_gnrl_pat', 'gate_sep_hp',
                             'gate_sep_hp_fw', 'fxec_gate_sep_hp', 
                             'leg_mask_fxec_gate_sep_hp', 'dg_ca3_gate_sep_hp',
                             'dg_dynca3_gate_sep_hp', 'dg_dynca3ca1_gate_sep_hp',
                             ]:
            if self.pass_states is not None:
                if self.pass_states.startswith('noisy_pass:') \
                        and (self.last_states is not None):
                    noise_norm = float(self.pass_states.split(':')[1])
                    noise = torch.randn(*self.last_states.shape).cuda() \
                            * noise_norm
                    self.last_states += noise
            rnn_output, all_states = self.rnn(input_embd, self.last_states)
            if self.pass_states is not None:
                if self.pass_states == 'pass_last' \
                        or self.pass_states.startswith('noisy_pass:'):
                    self.last_states = all_states[-1]
                else:
                    raise NotImplementedError
        elif self.rnn_type.startswith('parallel_hopfield_'):
            all_outputs = []
            for _rnn in self.rnn:
                all_outputs.append(_rnn(input_embd))
            rnn_output = torch.cat(all_outputs, dim=-1)
        elif self.rnn_type == 'hopfield':
            assert self.rnn_tile is None
            _inp = torch.transpose(input_embd, 0, 1)
            _inp_R = _inp[:, :-1, :]
            _inp_Y = _inp[:, -1:, :]
            
            rnn_output = self.rnn((_inp_Y, _inp_R, _inp_Y))
            rnn_output = torch.transpose(rnn_output, 0, 1)
        elif self.rnn_type in ['stm_aha']:
            rnn_output = self.rnn(input_embd)
            # return rnn_output
        else:
            raise NotImplemenntedError
            
        if isinstance(rnn_output, tuple):
            rnn_output, fastweight = rnn_output
            pred_output = self.pred_mlp([rnn_output[-1]], fastweight)[0]
        else:
            pred_output = self.pred_mlp([rnn_output[-1]])[0]

        return pred_output

@HEADS.register_module
class MSTHead(HippRNNHead):
    def __init__(
            self, rnn_kwargs,
            rnn_type='default', rnn_tile=None, pred_mlp = None):
        super(HippRNNHead, self).__init__()
        self.rnn_tile = rnn_tile
        
        self.rnn_type = rnn_type
        self.rnn_kwargs = rnn_kwargs
        self.pred_mlp_kwargs = pred_mlp
        self.build_rnn()
        
    def init_weights(self, init_linear='kaiming'):
        if self.pred_mlp_kwargs:
            self.pred_mlp.init_weights(init_linear=init_linear) 
        
    def build_rnn(self):
        if self.rnn_type == 'fastweights':
            self.rnn = rnn_modules.FastWeights(**self.rnn_kwargs)
        elif self.rnn_type == 'selflstm_mst':
            self.rnn = rnn_modules.SelfLSTM_MST(**self.rnn_kwargs)
        elif self.rnn_type == 'gate_sep_hp_lstm_readout_mst':
            self.rnn = rnn_modules.GateSepHPathRNN_LSTMreadout_MST(**self.rnn_kwargs)
        elif self.rnn_type in ['gate_sep_hp_mlp_readout_patresp_mst', 'gate_sep_hp_mlp_readout_rawpatresp_mst']:
            self.rnn = rnn_modules.GateSepHPathRNN_MLPreadout_patresp_MST(**self.rnn_kwargs)
            self.pred_mlp = builder.build_neck(self.pred_mlp_kwargs)
            
    def rnn_forward_neck(self, input_embd, output):
        """
            Assume output has shape [num_seq, batch_size, feature_dim]
        """
        rnn_output = []
        num_seq = input_embd.size(0)
        for idx in range(num_seq):
            rnn_output.append(self.pred_mlp([output[idx]])[0])
        rnn_output = torch.stack(rnn_output, axis=0)
        return rnn_output
    
    def forward(self, input_embd):
        # input_embd shape: [seq_len, batch_size, input_size]
        if self.rnn_tile is not None:
            input_embd = self.repeat_inputs(input_embd)

        if self.rnn_type in ['gate_sep_hp_lstm_readout_mst']:
            rnn_output, _ = self.rnn(input_embd)
        elif self.rnn_type in ['gate_sep_hp_mlp_readout_patresp_mst']:
            raw_pat_resps, _, _ = self.rnn(input_embd)
            rnn_output = self.rnn_forward_neck(input_embd, raw_pat_resps)
            return rnn_output, raw_pat_resps
        elif self.rnn_type in ['gate_sep_hp_mlp_readout_rawpatresp_mst']:
            _, pat_resps, _ = self.rnn(input_embd)
            rnn_output = self.rnn_forward_neck(input_embd, pat_resps)
            return rnn_output, pat_resps
        else:
            rnn_output = self.rnn(input_embd)
            
        return rnn_output

@HEADS.register_module
class ClassicalHopfield(nn.Module):
    def __init__(
            self, seq_len, pred_mlp, rnn_kwargs,
            exp = 2,
            rnn_type='default', rnn_tile=None):
        super().__init__()
        self.rnn_tile = rnn_tile
        self.seq_len = seq_len
        self.pred_mlp = builder.build_neck(pred_mlp)
        self.rnn_type = rnn_type
        self.rnn_kwargs = rnn_kwargs
        self.exp = exp
        print("self.exp",self.exp)
        # self.build_rnn()
        
    
        
    def forward(self, input_embd, n_iter=15):
        """
        input_embd: Tensor of shape [9, 256, 128]
            sequence_vectors = input_embd[:8, 256, 128] holds the sequence
            noisy_input = input_embd[8, 256, 128]  holds the corrupted cue item
        diffs: shape [8, 256]
        cleaned: shape [256,128]
        """
        def clean_up_guess(cleaned):
            #print("Run cleanup")
            cleaned = cleaned.requires_grad_()
            optimizer = torch.optim.Adam([cleaned], lr=0.01)
            for step in range(n_iter):
                optimizer.zero_grad()
                cleaned_normed = F.normalize(cleaned, dim=1)
                diffs = torch.einsum('bf,sbf->sb', cleaned_normed, sequence_vectors)
                diffs = torch.pow(diffs, self.exp)
                diffs = -torch.sum(diffs)
#                 diffs.backward(retain_graph=True)
                diffs.backward(retain_graph=True)
                #print(diffs)
                optimizer.step()
                #print("cleaned-target", diffs, torch.mean(torch.norm(cleaned_normed-target,dim=0,p=2)))
                #print("target", target.shape, target[0,:10])
                #print("cleaned", cleaned_normed.shape, cleaned_normed[0,:10])
                #alt = torch.einsum('bf,sbf->sb', cleaned_normed, sequence_vectors)
                #alt = torch.pow(alt, self.exp)
                #print("step, alt", step, alt.shape, alt[:,:10])
            return cleaned_normed.detach()
        sequence_vectors = input_embd[:self.seq_len]
        noisy_input = input_embd[self.seq_len]#.unsqueeze(0)
#         for batch in range(sequence_vectors.size(1)):
#             cleaned = noisy_input[0, batch, :].detach().clone()
        return_array = []
        return_array.append(clean_up_guess(noisy_input.detach().clone()))
        for _ in range(3):
            #print("Run cleanup")
#                 cleaned_normed = clean_up_guess(F.normalize(torch.randn_like(noisy_input), dim=1))
            return_array.append(clean_up_guess(torch.randn_like(noisy_input)))
            #return_array.append(F.normalize(torch.randn_like(noisy_input), dim=1))
        # Loss[Seq, Batch] = Query[Batch, Feat] * Key[Seq, Batch, Feat]
            
        #diffs = (noisy_input - sequence_vectors)
        #diffs = torch.norm(diffs, dim = 2, p = 2)
        #print(diffs[:,:10], diffs.shape)
        #print(torch.argsort(diffs,dim=0)[:,:10])
        
        #self.hopfield_encode(sequence_vectors)
        #self.hopfield_retrieve(noisy_input)
        return_tensor = torch.cat(return_array, dim=1)
        return_tensor.requires_grad = False
        return return_tensor
    
    def init_weights(self, init_linear='kaiming'):
        self.pred_mlp.init_weights(init_linear=init_linear)

@HEADS.register_module
class HippMLPHead(nn.Module):
    def __init__(
            self, pred_mlp):
        super().__init__()
        self.pred_mlp = builder.build_neck(pred_mlp)

    def init_weights(self, init_linear='kaiming'):
        self.pred_mlp.init_weights(init_linear=init_linear)

    
    def forward(self, input_embd):
        # input_embd shape: [seq_len, batch_size, input_size]
        new_input_embd = input_embd.transpose(0, 1)
        new_input_embd = new_input_embd.reshape(new_input_embd.size(0), -1)
        pred_output = self.pred_mlp([new_input_embd])[0]
        return pred_output
