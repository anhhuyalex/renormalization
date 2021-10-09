import copy
import os
from . import sy_embd_pat_sep
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')


def mmax_inner_func(cfg, hidden_size=256):
    cfg.data['train'] = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': 32,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_train_meta.txt'),
            }
    cfg.model = {
            'type': 'HippRNN',
            'seq_len': 32,
            'num_negs': 3,
            'noise_norm': None,
            'loss_type': 'just_pos',
            'mask_use_kNN': True,
            'hipp_head': dict(
                type='HippRNNHead',
                rnn_type='gnrl_pat',
                rnn_tile=3,
                rnn_kwargs=dict(
                    input_size=128, 
                    hidden_size=hidden_size,
                    naive_hidden=False, 
                    hand_coded_softmax=False,
                    pattern_size=100),
                pred_mlp=dict(
                    type='NonLinearNeckV1',
                    in_channels=hidden_size, hid_channels=hidden_size,
                    out_channels=512, with_avg_pool=False),
                ),
            }
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


def simclr_embd_ctl(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def sc_fast_weights(cfg):
    cfg = simclr_embd_ctl(cfg)
    cfg.model['hipp_head']['rnn_kwargs'] = {
            'input_size': 128,
            'hidden_size': 512,
            'num_layers': 1,
            'update_fast_weight' : 'resnet_1_layer_cx_hx_separate'
            }
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.model['hipp_head']['rnn_type'] = 'fastweights'
    cfg.model['hipp_head']['pred_mlp'] = {
            'type': 'NonLinearNeckFW',
            'in_channels': 512,
            'out_channels': 512,
            'with_avg_pool': False}
    return cfg


def simclr_p200_embd_ctl(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 64
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    return cfg


def sc_p200_ctl_notile(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 128
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_tile'] = None
    return cfg


def sc_p200_ctl_layernorm(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 64
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_kwargs']['with_layernorm'] = True
    cfg.model['hipp_head']['pred_mlp'] = {
            'type': 'NonLinearNeckV3',
            'in_channels': 256,
            'hid_channels': 256,
            'out_channels': 512,
            'with_avg_pool': False}
    return cfg


def sc_p200_ctl_tanh(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 64
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    return cfg


def simclr_p200_embd_ctl_sl(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0001)
    return cfg


def simclr_p200_embd_ctl_ft(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-4, weight_decay=0.0001)
    return cfg


def simclr_p200_embd_ctl_slnwd(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    return cfg


def sc_p200_slnwd_tanh(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    return cfg


def sc_p200_slnwd_tanh_notile(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def sc_p200_sl_nt_sepmlp(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5, weight_decay=1e-4)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.data['imgs_per_gpu'] = 128
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    return cfg


def slnwd_sm_nt_setting(cfg):
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    return cfg


def sc_p200_slnwd_nt_sepmlp(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def sc_p200_sl_tanh_nt_sepmlp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5, weight_decay=1e-4)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.data['imgs_per_gpu'] = 128
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    return cfg


def sc_p200_slnwd_tanh_notile_sepmlp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def seq64_setting(cfg):
    cfg.data['imgs_per_gpu'] = 64
    cfg.data['train']['seq_len'] = 64
    cfg.model['seq_len'] = 64
    return cfg


def sc_p200_seq64_sl_nt_sepmlp(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5, weight_decay=1e-4)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    cfg = seq64_setting(cfg)
    return cfg


def sc_p200_seq64_slnwd_tanh_nt_sepmlp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    return cfg

def sc_p200_seq96_slnwd_tanh_nt_sepmlp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.data['train']['seq_len'] = 96
    cfg.model['seq_len'] = 96
    cfg.data['imgs_per_gpu'] = 32
    return cfg


def sc_seq64_allrl_sm(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    return cfg


def sc_seq64_gatestp_all(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            'act_func': 'tanh',
            }
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_mlp_stpmlp'
    #cfg.data['imgs_per_gpu'] = 32
    cfg.data['imgs_per_gpu'] = 64
    return cfg

def sc_seq96_gatestp_all(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.data['train']['seq_len'] = 96
    cfg.model['seq_len'] = 96
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            'act_func': 'tanh',
            }
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_mlp_stpmlp'
    return cfg


def sc_seq64_notile_sm(cfg):
    cfg = simclr_p200_embd_ctl(cfg)
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    cfg = seq64_setting(cfg)
    return cfg


def sc_seq64_all_sm_sim(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)

    cfg.model['hipp_head']['rnn_type'] = 'sim_gnrl_pat'
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    return cfg


def sc_seq64_all_sm_mltmx(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)

    cfg.model['hipp_head']['rnn_type'] = 'mltmx_gnrl_pat'
    return cfg


def sc_seq64_all_sm_mxnm(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'max'
    return cfg


def sc_seq64_all_sm_sgd(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.optimizer = dict(type='SGD', lr=1e-3, momentum=0.9)
    return cfg


def sc_seq64_all_sm_ftsgd(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.optimizer = dict(type='SGD', lr=1e-2, momentum=0.9)
    cfg.lr_config = dict(policy='step', step=[66, 78])
    return cfg


def sc_seq64_all_sm_ftsgd_cos(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.optimizer = dict(type='SGD', lr=1e-2, momentum=0.9)
    cfg.lr_config = dict(
        policy='CosineAnnealing',
        min_lr=0.,
        warmup='linear',
        warmup_iters=3,
        warmup_ratio=0.0001, # cannot be 0
        warmup_by_epoch=True)
    return cfg


def sc_seq64_all_sm_fftsgd(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.optimizer = dict(type='SGD', lr=1e-1, momentum=0.9)
    return cfg


def sc_seq64_all_sm_ftAdam(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)

    cfg.optimizer = dict(type='Adam', lr=5e-4)
    return cfg


def sc_seq64_all_sm_gate_2mlps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    return cfg


def sc_seq64_all_sm_gate_2mlps_ctmx(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'concat_max'
    return cfg


def sc_seq64_all_sm_ctmx(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'concat_max'
    return cfg


def ctmx_sephp_setting(cfg):
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'concat_max'
    cfg.model['hipp_head']['rnn_type'] = 'gate_sep_hp'
    return cfg


def sc_seq64_all_sm_gate_2mlps_ctmx_sephp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.total_epochs = 400
    return cfg


def sc_seq64_all_sm_sim_gate_ctmx_sephp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.total_epochs = 400
    return cfg


def sc_seq64_new_sim_gate_ctmx_ps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['pass_states'] = 'pass_last'
    cfg.total_epochs = 400
    return cfg


def sc_seq64_new_sim_gate_ctmx_rawps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['pass_states'] = 'pass_last'
    cfg.model['hipp_head']['rnn_kwargs']['state_pass_method'] = 'raw_pass'
    cfg.total_epochs = 800
    return cfg


def sc_seq64_new_sim_gate_ctmx_nsps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['pass_states'] = 'noisy_pass:0.1'
    cfg.total_epochs = 400
    return cfg


def sc_seq64_new_sim_gate_ctmx_ns2ps(cfg):
    cfg = sc_seq64_new_sim_gate_ctmx_nsps(cfg)
    cfg.model['hipp_head']['pass_states'] = 'noisy_pass:0.2'
    return cfg


def sc_seq64_all_sm_sim_gate_sephp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_type'] = 'gate_sep_hp'
    cfg.total_epochs = 400
    return cfg


def sc_seq64_all_sm_gate_2mlps_ctmx_sephp_cr(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['loss_type'] = 'pos_with_corr'
    return cfg


def sc_seq64_all_sm_gate_2mlps_sephp(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.total_epochs = 400
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    cfg.model['hipp_head']['rnn_type'] = 'gate_sep_hp'
    return cfg


def sc_seq64_all_sm_sim_gate_2mlps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    return cfg


def seq96_setting(cfg):
    cfg.data['imgs_per_gpu'] = 64
    cfg.data['train']['seq_len'] = 96
    cfg.model['seq_len'] = 96
    return cfg


def sc_seq96_new_sim_gate(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq96_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.total_epochs = 400
    return cfg


def varylen_setting(cfg):
    cfg.data['imgs_per_gpu'] = 64
    cfg.data['train']['seq_len'] = 96
    cfg.data['train']['min_seq_len'] = 32
    cfg.data['train']['batch_size'] = 256
    cfg.data['train']['type'] = 'VaryLenSAYCamSeqVec'
    cfg.model['seq_len'] = None
    return cfg


def sc_varylen_new_sim_gate_ctmx(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_mstft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.data['train']['which_model'] = 'simclr_mst_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_pairft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_rth_pft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 800
    return cfg


def sc_varylen_new_sim_gate_ctmx_rth_pft_rps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.model['hipp_head']['pass_states'] = 'pass_last'
    cfg.model['hipp_head']['rnn_kwargs']['state_pass_method'] = 'raw_pass'
    cfg.total_epochs = 800
    return cfg


def sc_varylen_new_sim_gate_resmlp_rth_pft_rps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th'
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'res_mlp'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.model['hipp_head']['pass_states'] = 'pass_last'
    cfg.model['hipp_head']['rnn_kwargs']['state_pass_method'] = 'raw_pass'
    cfg.total_epochs = 800
    return cfg


def pft_reluth_samps(cfg):
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.model['hipp_head']['pass_states'] = 'pass_last'
    cfg.model['hipp_head']['rnn_kwargs']['state_pass_method'] = 'sample_from_q'
    return cfg


def sc_pft_vlen_reluth_samps(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.total_epochs = 800
    return cfg


def sc_pft_vlen_reluth_samps_p400(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 400
    cfg.total_epochs = 800
    return cfg


def sc_pft_vlen64_reluth(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.data['train']['seq_len'] = 64
    cfg.total_epochs = 800
    return cfg


def typical_hlf(cfg, ratio=2):
    cfg.data['train']['sub_dim'] = 128 // ratio
    cfg.model['hipp_head']['rnn_kwargs']['input_size'] = 128  // ratio
    cfg.model['hipp_head']['rnn_kwargs']['hidden_size'] = 256 // ratio
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 256 // ratio
    cfg.model['hipp_head']['pred_mlp']['hid_channels'] = 256 // ratio
    cfg.model['hipp_head']['pred_mlp']['out_channels'] = 512 // ratio
    return cfg


def sc_pft_vlen64_reluth_hlf(cfg):
    cfg = sc_pft_vlen64_reluth(cfg)
    cfg = typical_hlf(cfg)
    cfg.data['train']['batch_size'] = 512
    return cfg


def sc_pft_vlen64_reluth_4th(cfg):
    cfg = sc_pft_vlen64_reluth(cfg)
    cfg = typical_hlf(cfg, 4)
    cfg.data['train']['batch_size'] = 512
    return cfg


def sc_pft_vlen64_reluth_samps_p300(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 300
    cfg.data['train']['seq_len'] = 64
    cfg.total_epochs = 800
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 300
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.15/200
    cfg.data['train']['seq_len'] = 64
    cfg.total_epochs = 800
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_lm(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'leg_mask_fxec_gate_sep_hp'
    cfg.model['use_leg_mask'] = True
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_op(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['only_process_valid_input'] = True
    cfg.model['hipp_head']['rnn_kwargs']['sync_update'] = False
    return cfg


def add_ca3_dg_tr_cfgs(cfg):
    ca3_cfg = dict(
            type='PatSepRecNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_MLP_CFG),
            pr_neck=copy.deepcopy(sy_embd_pat_sep.TYPICAL_PR),
            pretrained=None,
            )
    dg_cfg = dict(
            type='PatSepRecNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_S30_WW_CFG),
            pr_neck=copy.deepcopy(sy_embd_pat_sep.TYPICAL_PR),
            pretrained=None,
            )
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = ca3_cfg
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = dg_cfg
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    #cfg.model['only_process_valid_input'] = True
    #cfg.model['hipp_head']['rnn_kwargs']['sync_update'] = False
    cfg = add_ca3_dg_tr_cfgs(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'dg_ca3_gate_sep_hp'
    cfg.model['include_additional_loss'] = True
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3_unl(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = add_ca3_dg_tr_cfgs(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'dg_ca3_gate_sep_hp'
    return cfg


def dgdynca3_func(cfg):
    cfg.model['hipp_head']['rnn_type'] = 'dg_dynca3_gate_sep_hp'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.DynamicMLP2L_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.NAIVE_S30_WW_CFG)
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_fl(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 1e-2
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_ffl(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 1e-1
    return cfg


def sc_pft_vlen64_p300n_dgdynca3_step4(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'NStepDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['step_num'] = 4
    return cfg


def sc_pft_vlen64_p300n_dgdynca3_step8(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'NStepDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['step_num'] = 8
    return cfg


def sc_pft_vlen64_p300n_dgdynca3_rep8(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'RepDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['rep_num'] = 8
    return cfg


def sc_pft_vlen64_p300n_dgdynca3_rep4(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'RepDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['rep_num'] = 4
    return cfg


def sc_pft_vlen64_p300n_dgdynca3_s8r16(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'NStepRepDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['rep_num'] = 16
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['step_num'] = 8
    return cfg


def dgdynca3ca1_func(cfg):
    cfg.model['hipp_head']['rnn_type'] = 'dg_dynca3ca1_gate_sep_hp'
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.DynamicMLP2L_S8R16_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.NAIVE_S30_WW_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca1_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.NAIVE_S60_W_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.DynamicMLP2L_S8R16_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['in_channels'] = 256
    cfg.model['hipp_head']['rnn_kwargs']['readout_mlp_cfg'] = dict(
            type='NonLinearNeckV1',
            in_channels=256, hid_channels=256,
            out_channels=128, with_avg_pool=False)
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['num_negs'] = 0
    cfg.model['include_additional_loss'] = True
    return cfg


def sc_pft_vlen64_p300n_dgdynca3ca1(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_func(cfg)
    cfg.data['train']['batch_size'] = 48 * 4
    cfg.data['train']['data_len'] = 256 * 5000 - 128
    return cfg

def sc_pft_p300n_dgdynca3ca1(cfg):
    cfg = sc_pft_vlen64_p300n_dgdynca3ca1(cfg)
    cfg.data['train'] = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': 32,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_train_meta.txt'),
            }
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.data['workers_per_gpu'] = 2
    cfg.data['data_len'] = 64
    
    
    return cfg

def sc_pft_p300n_dgdynca3ca1_seq64(cfg):
    cfg = sc_pft_p300n_dgdynca3ca1(cfg)
    cfg.data['train']['seq_len'] = 64
    cfg.data['imgs_per_gpu'] = 48
    return cfg

def sc_pft_p300n_dgdynca3ca1_seq96(cfg):
    cfg = sc_pft_p300n_dgdynca3ca1(cfg)
    cfg.data['train']['seq_len'] = 96
    cfg.data['imgs_per_gpu'] = 36
    return cfg

def sc_pft_p300n_aha_dgca3(cfg):
    cfg = sc_pft_p300n_dgdynca3ca1(cfg)
    cfg.data['train']['seq_len'] = 32
    cfg.model['hipp_head'] = dict(
                type='HippRNNHead',
                rnn_type='stm_aha',
                rnn_tile=None,
                rnn_kwargs=dict(
                    input_size=128, 
                    dg_cfg = {#'type': 'AHA_DG_Sparse_LinearMLP', 
                                'type': 'AHA_linearDG_kSparseInhibition', 
                                'in_channels': 128, 
                                'out_channels': 225, 
                                'sparsity': 30,
                                'knockout_rate': 0.25,
                                'init_scale': 10.0,
                                'inhibition_decay': 0.95
                                },
                    perforant_cfg = dict(
                        type = "PerforantHebb_AHA",
                        ec_shape = 128, 
                        dg_shape = 225, 
                        ca3_shape = 225, 
                        learning_rate = 0.01,
                        reset_params = True, 
                        reset_optim = True,
                        use_dg_ca3 = False,
                    ),
                    ca3_cfg = dict(
                        type = "KNNBuffer_AHA",
                        input_shape = 225, 
                        target_shape = 225, 
                        shift_range = False
                    ),
                    msp_cfg = dict(
                        type = "MonosynapticPathway_AHA",
                        ca3_shape = 225, 
                        ec_shape = 128, 
                        ca1_reset_params = False,
                        ca1_reset_optim = False,
                        ca3_ca1_reset_params = True,
                        ca3_ca1_reset_optim = True,
                        ca3_recall = True,
                        ca1_cfg = dict( 
                          learning_rate= 0.001,
                          weight_decay= 0.000025,

                          hidden_size= 800,
                          input_dropout= 0.0,
                          hidden_dropout= 0.0,

                          encoder_nonlinearity= "leaky_relu",
                          decoder_nonlinearity= "leaky_relu",
                          use_bias= True,
                          norm_inputs= False,
                        ),
                        ca3_ca1_cfg= dict( 
                            learning_rate= 0.01,
                            weight_decay= 0.00004,
                            
                            hidden_size= 100,
                            input_dropout= 0.0,
                            hidden_dropout= 0.0,
                            
                            encoder_nonlinearity= "leaky_relu",
                            decoder_nonlinearity= "leaky_relu",
                            use_bias= True,
                            norm_inputs= True
                        )
                    ),
                    is_hebbian_perforant=True,
                    hidden_size=256,
                    naive_hidden=False, 
                    hand_coded_softmax=False,
                    pattern_size=100),
                pred_mlp=dict(
                    type='Identity'
#                     type='NonLinearNeckV1',
#                     in_channels=256, 
#                     hid_channels=256,
#                     out_channels=512, 
#                     with_avg_pool=False
                    ),
    )
    cfg.model["include_additional_loss"] = False
    print("cfg", cfg)
    return cfg

def sc_pft_vlen64_p300n_dgdynca3ca1_osfilter(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_func(cfg)
    #cfg.data['train']['batch_size'] = 48 * 4
    #cfg.data['train']['data_len'] = 256 * 5000 - 128
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    return cfg


TOP1MX_CFG = dict(
        type='ScaleTopK',
        sparsity=1,
        scale=10,
        )
def sc_pft_vlen64_p300n_dgdynca3ca1_top1mx(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_func(cfg)
    cfg.data['train']['batch_size'] = 48 * 4
    cfg.data['train']['data_len'] = 256 * 5000 - 128
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = TOP1MX_CFG
    return cfg


def sc_pft_vlen64_p300n_dynca1_top1mx_smq(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = TOP1MX_CFG
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 1024
    return cfg


def dgdynca3ca1_r32_func(cfg):
    cfg = dgdynca3ca1_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.DynamicMLP2L_R32_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['in_channels'] = 256
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.DynamicMLP2L_R32_CFG)
    return cfg


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 1024
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    return cfg


def make_cfg_hlf_chnnls(cfg_dict, ratio):
    cfg_dict['in_channels'] = cfg_dict['in_channels'] // ratio
    cfg_dict['hid_channels'] = cfg_dict['hid_channels'] // ratio
    cfg_dict['out_channels'] = cfg_dict['out_channels'] // ratio
    if 'sparsity' in cfg_dict:
        cfg_dict['sparsity'] = cfg_dict['sparsity'] // ratio
    return cfg_dict


def ca1_hlf(cfg, ratio=2):
    cfg.data['train']['sub_dim'] = 128 // ratio
    cfg.model['hipp_head']['rnn_kwargs']['input_size'] = 128 // ratio
    cfg.model['hipp_head']['rnn_kwargs']['hidden_size'] = 256 // ratio
    cfg.model['hipp_head']['rnn_kwargs']['ca1_cfg'] = make_cfg_hlf_chnnls(
            cfg.model['hipp_head']['rnn_kwargs']['ca1_cfg'], ratio)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = make_cfg_hlf_chnnls(
            cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'], ratio)
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = make_cfg_hlf_chnnls(
            cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'], ratio)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg'] = make_cfg_hlf_chnnls(
            cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg'], ratio)
    cfg.model['hipp_head']['rnn_kwargs']['readout_mlp_cfg'] = make_cfg_hlf_chnnls(
            cfg.model['hipp_head']['rnn_kwargs']['readout_mlp_cfg'], ratio)
    return cfg


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 1024
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    cfg = ca1_hlf(cfg)
    cfg.data['train']['batch_size'] = 512
    return cfg


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf_prcmp(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 1024
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    cfg = ca1_hlf(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['precomp_dg_ca1'] = True
    cfg.data['train']['batch_size'] = 512
    return cfg


def sc_pft_vlen64_p300n_dynca1_sg_osf_smq_hlf_prcmp(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 2048
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['precomp_dg_ca1'] = True
    cfg.data['train']['batch_size'] = 512
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 1024
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    cfg = ca1_hlf(cfg, 4)
    cfg.data['train']['batch_size'] = 512
    return cfg


def osf_smq_4th_prcmp(cfg):
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.model['hipp_head']['rnn_kwargs']['pt_queue_len'] = 4096
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    cfg.model['hipp_head']['rnn_kwargs']['precomp_dg_ca1'] = True
    cfg.data['train']['batch_size'] = 1024
    cfg.data['imgs_per_gpu'] = 256

    cfg.data['train']['cache_folder'] = os.path.join(
            SY_DATASET_DIR,
            'infant_headcam/embeddings_related/',
            cfg.data['train']['which_model'],
            'seq_{}_{}_bs_{}'.format(
                cfg.data['train']['min_seq_len'], 
                cfg.data['train']['seq_len'],
                cfg.data['train']['batch_size'],
                ))
    return cfg


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = ca1_hlf(cfg, 4)
    return cfg


def selfgrad_func(cfg):
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.SelfGradDynamicMLP2L_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg'] = copy.deepcopy(
            sy_embd_pat_sep.SelfGradDynamicMLP2L_CFG)
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['in_channels'] = 256
    cfg.model['hipp_head']['rnn_kwargs']['update_in_fwd'] = True
    return cfg


def sc_pft_vlen64_p300n_dynca1_sg_osf_smq_4th_prcmp(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    return cfg


def sc_pft_vlen64_p300n_dynca1_4th_typ_slw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 5e-4
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 5e-4
    return cfg


def sc_pft_vlen64_p300n_dynca1_4th_typ_sslw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 1e-4
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 1e-4
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_ll(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.4/200
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_l(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.3/200
    return cfg


def setup_filter_dataset(cfg):
    cfg.data['train']['type'] = 'FilterVLenSCSeqVec'
    cfg.data['train']['cache_folder'] = os.path.join(
            SY_DATASET_DIR,
            'infant_headcam/embeddings_related/',
            cfg.data['train']['which_model'],
            'seq_{}_{}'.format(
                cfg.data['train']['min_seq_len'], 
                cfg.data['train']['seq_len']))
    return cfg


def sc_pft_vlen64_reluth_p300sn_l_filter(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.3/200
    cfg = setup_filter_dataset(cfg)
    return cfg


def sc_pft_vlen64_reluth_p300sn_l_filter_fst(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.3/200
    cfg = setup_filter_dataset(cfg)
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    return cfg


def sc_pft_vlen64_reluth_p300sn_l_osfilter(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.3/200
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    return cfg


def sc_pft_vlen64_reluth_p300sn_l_osfilter_fst(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.3/200
    cfg.data['train']['type'] = 'OSFilterVLenSCSeqVec'
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_s(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.1/200
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_fw(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 300
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.15/200
    cfg.data['train']['seq_len'] = 64
    cfg.model['hipp_head']['rnn_type'] = 'gate_sep_hp_fw'
    cfg.model['hipp_head']['pred_mlp']['type'] = 'NonLinearNeckV1FW'
    cfg.total_epochs = 800
    return cfg


def add_ca3_dg_cfgs(cfg):
    MODEL_CKPT_DIR = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/'
    ca3_cfg = dict(
            type='PatSepNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_MLP_CFG),
            pretrained=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_dg_2/latest_cached.pth'),
            )
    dg_cfg = dict(
            type='PatSepNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_S30_WW_CFG),
            pretrained=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_s30_ww_dg/latest_cached.pth'),
            )
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = ca3_cfg
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = dg_cfg
    return cfg


def sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 300
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.15/200
    cfg.model['hipp_head']['rnn_type'] = 'fxec_gate_sep_hp'
    cfg = add_ca3_dg_cfgs(cfg)
    cfg.data['train']['seq_len'] = 64
    cfg.total_epochs = 800
    return cfg


def add_ca3_dg_rec_cfgs(cfg):
    MODEL_CKPT_DIR = '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/'
    ca3_cfg = dict(
            type='PatSepRecNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_MLP_CFG),
            pr_neck=copy.deepcopy(sy_embd_pat_sep.TYPICAL_PR),
            pretrained=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_dg_rec/epoch_40.pth'),
            )
    dg_cfg = dict(
            type='PatSepRecNoGrad',
            ps_neck=copy.deepcopy(sy_embd_pat_sep.NAIVE_S30_WW_CFG),
            pr_neck=copy.deepcopy(sy_embd_pat_sep.WW_PR),
            pretrained=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_s30_ww_dg_rec_2/epoch_40.pth'),
            )
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg'] = ca3_cfg
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg'] = dg_cfg
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(cfg)
    cfg = add_ca3_dg_rec_cfgs(cfg)
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_tr(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(cfg)
    cfg = add_ca3_dg_rec_cfgs(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['fix_ca3_dg_weights'] = False
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['with_grad'] = True
    cfg.model['hipp_head']['rnn_kwargs']['dg_cfg']['with_grad'] = True
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_wc(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(cfg)
    cfg = add_ca3_dg_rec_cfgs(cfg)
    cfg.model['loss_type'] = 'pos_with_corr'
    return cfg


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_lm(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(cfg)
    cfg = add_ca3_dg_rec_cfgs(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'leg_mask_fxec_gate_sep_hp'
    cfg.model['use_leg_mask'] = True
    return cfg


def sc_pft_vlen_reluth_samps_p400_nsdcy(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg = pft_reluth_samps(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 400
    cfg.model['hipp_head']['rnn_kwargs']['noisy_decay'] = 0.15/200
    cfg.total_epochs = 800
    return cfg


def sc_varylen_new_sim_gate_ctmx_mx_pft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'cat_max10'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_mxl4_pft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['max_mlp_layers'] = 4
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_mlp2_pft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'mlp2'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_new_sim_gate_ctmx_rthmx_pft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['pr_gate_method'] = 'relu_th_cat_max10'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_seq64_new_sim_gate_ctmx_pairft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = seq64_setting(cfg)
    cfg = ctmx_sephp_setting(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'sim_gate_two_mlps'
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.total_epochs = 400
    return cfg


def sc_varylen_naive_pairft(cfg):
    cfg = sc_p200_ctl_tanh(cfg)
    cfg = slnwd_sm_nt_setting(cfg)
    cfg = varylen_setting(cfg)
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['max_norm'] = 'concat_max'
    cfg.total_epochs = 400
    return cfg


def sc_pft_varylen_fast_weights(cfg):
    cfg = sc_fast_weights(cfg)
    cfg = varylen_setting(cfg)
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    return cfg


def sc_p200_gate_2mlps(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    return cfg


def sc_p200_gate_2mlps_newh_pm(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 64
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_two_mlps'
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    return cfg


def simclr16_embd_ctl(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['train']['seq_len'] = 16
    cfg.model['seq_len'] = 16
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def simclr_embd_stpmlp(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    return cfg


def gate_stpmlp_setting(cfg):
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            }
    return cfg


def simclr_embd_gate_stpmlp(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 32
    cfg = gate_stpmlp_setting(cfg)
    return cfg


def sc_p200_gstpmlp_newh_pm(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'simclr_sy'
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.model['hipp_head']['rnn_kwargs']['newh_from_parallel_mmax'] = True
    cfg = gate_stpmlp_setting(cfg)
    return cfg


def sc_p200_gstpmlp_newh_pm_nwd(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg.optimizer = dict(type='Adam', lr=1e-4)
    return cfg


def sc_p200_gstpmlp_newh_pm_sl_nwd(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    return cfg


def sc_p200_sepmlp_slnwd_tanh(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            'act_func': 'tanh',
            }
    return cfg


def sc_p200_sepmlp_slnwd_tanh_notile(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            'act_func': 'tanh',
            }
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.data['imgs_per_gpu'] = 64
    return cfg


def sc_p200_gate_sepmlp_slnwd_tanh_nt(cfg):
    cfg = sc_p200_gstpmlp_newh_pm(cfg)
    cfg.optimizer = dict(type='Adam', lr=5e-5)
    cfg.model['hipp_head']['rnn_kwargs']['act_func'] = 'tanh'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            'act_func': 'tanh',
            }
    cfg.model['hipp_head']['rnn_tile'] = None
    cfg.data['imgs_per_gpu'] = 64
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'gate_mlp_stpmlp'
    return cfg


def moco_embd_ctl(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'moco_sy'
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def moco16_embd_ctl(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.data['train']['which_model'] = 'moco_sy'
    cfg.data['train']['seq_len'] = 16
    cfg.model['seq_len'] = 16
    cfg.data['imgs_per_gpu'] = 128
    return cfg
