import copy
import os
MST_DATASET_DIR = os.environ.get(
        'MST_DATASET_DIR',
        '/mnt/fs1/Dataset')
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')

def mst_synth_cfg_func(cfg, hidden_size=512):
    cfg.data['train'] = {
            'type': 'MSTSynthVectorDataset',
            'bank_size': 192,
            'root': MST_DATASET_DIR,
            'data_len': 256*5000,
            }
    cfg.model = {
            'type': 'MSTRNN',
            'seq_len': 190,
            'loss_type': 'default',
            'hipp_head': dict(
                type='MSTHead',
                rnn_type='selflstm_mst',
                rnn_kwargs=dict(
                    input_size=128, 
                    output_size = 3,
                    hidden_size=256, 
                    num_layers=2),
#                 pred_mlp=dict(
#                     type='NonLinearNeckV1',
#                     in_channels=hidden_size, hid_channels=hidden_size,
#                     out_channels=512, with_avg_pool=False),
                )
            }
    cfg.optimizer = dict(
            type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
    cfg.data['imgs_per_gpu'] = 256
    cfg.data['workers_per_gpu'] = 0
    return cfg

def mst_saycam_cfg_func(cfg, hidden_size=512):
    cfg = mst_synth_cfg_func(cfg, hidden_size)
    cfg.data['train'] = {
            'type': 'MSTSaycamVectorDataset',
            'bank_size': 192,
            'seq_len': 190,
            'saycam_root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'root': MST_DATASET_DIR,
            'which_model': 'simclr_sy',
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_train_meta.txt'),
            'data_len': 256*5000,
            }
    
    
    
    return cfg

def mst_saycam_readout_cfg_func(cfg, hidden_size=512):
    cfg = mst_synth_cfg_func(cfg, hidden_size)
    cfg.data['train'] = {
            'type': 'MSTSaycamVectorDataset',
            'bank_size': 192,
            'seq_len': 190,
            'saycam_root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'root': MST_DATASET_DIR,
            'which_model': 'simclr_sy',
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_train_meta.txt'),
            'data_len': 256*5000,
            }
    cfg.data['imgs_per_gpu'] = 32
    cfg.model = {
            'type': 'MSTRNN',
            'seq_len': 190,
            'loss_type': 'default',
            'outdir': 'results_alex/hipp_simclr/mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion/confusion',
            'hipp_head': dict(
                type='MSTHead',
                rnn_type='gate_sep_hp_lstm_readout_mst',
                rnn_tile= None, 
                rnn_kwargs=dict(
                    input_size=128, 
                    hidden_size=256,
                    naive_hidden = False,
                    hand_coded_softmax = False,
                    pattern_size = 200,
                    act_func = 'tanh',
                    newh_from_parallel_mmax = True,
                    gate_update = True,
                    use_stpmlp = 'sim_gate_two_mlps',
                    max_norm = 'concat_max',
                    readout_input_size = 256, 
                    readout_hidden_size = 3, 
                    readout_output_size = None, 
                    readout_num_layers = 1,
                    readout_activation = 'tanh',
                ),
                 pred_mlp=dict(
                     type='NonLinearNeckV1',
                     in_channels=256, hid_channels=256,
                     out_channels=512, with_avg_pool=False),
                )
            }

    
    return cfg

def mst_saycam_mlpreadout_patrep_cfg_func(cfg, hidden_size=512):
    cfg = mst_saycam_readout_cfg_func(cfg, hidden_size)
    cfg.data['imgs_per_gpu'] = 32
    cfg.model = {
            'type': 'MSTRNN',
            'seq_len': 190,
            'loss_type': 'default',
            'outdir': 'results_alex/hipp_simclr/mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion/confusion',
            'hipp_head': dict(
                type='MSTHead',
                rnn_type='gate_sep_hp_mlp_readout_patresp_mst',
                rnn_tile= None, 
                rnn_kwargs=dict(
                    input_size=128, 
                    hidden_size=256,
                    naive_hidden = False,
                    hand_coded_softmax = False,
                    pattern_size = 200,
                    act_func = 'tanh',
                    newh_from_parallel_mmax = True,
                    gate_update = True,
                    use_stpmlp = 'sim_gate_two_mlps',
                    max_norm = 'concat_max',
                ),
                 pred_mlp=dict(
                     type='NonLinearNeckV1',
                     in_channels=200, hid_channels=256,
                     out_channels=3, with_avg_pool=False),
                )
            }
    cfg.total_epochs = 600
    return cfg

def mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_patrep_cfg_func(cfg, hidden_size)
    cfg.data['imgs_per_gpu'] = 128
    cfg.model.hipp_head.pred_mlp = dict(
                     type='NonLinearNeckV3',
                     in_channels=200, hid_channels=256,
                     out_channels=3, with_avg_pool=False) 
    return cfg

def mst_saycam_mlpreadout_rawpatrep_cfg_func(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_patrep_cfg_func(cfg, hidden_size)
    cfg.model.hipp_head.rnn_type = 'gate_sep_hp_mlp_readout_rawpatresp_mst'
    return cfg

def mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3(cfg, hidden_size)
    cfg.model.hipp_head.rnn_type = 'gate_sep_hp_mlp_readout_rawpatresp_mst'
    return cfg

def mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3(cfg, hidden_size)
    return cfg

def mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3(cfg, hidden_size)
    return cfg

def mst_saycam_val_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion(cfg, hidden_size=512):
    cfg = mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion(cfg, hidden_size)
    cfg.data.train['data_len'] = 500
    return cfg