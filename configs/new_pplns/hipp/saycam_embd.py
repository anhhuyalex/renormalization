from ..saycam_embd_param_setter import SAYCamEmbdParamBuilder
import openselfsup.config_related.sy_embd_hipp_funcs as sy_embd_hipp_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_rnn_params(
        args, exp_id, cfg_func, vary_len_val=None, **kwargs):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamEmbdParamBuilder(
            args=args, exp_id=exp_id, 
            cfg_path=cfg_path,
            vary_len_val=vary_len_val,
            add_svm_val=True, col_name='hipp_sy_embd',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            **kwargs)
    params = param_builder.build_params()
    return params


def simclr_embd_ctl(args):
    return get_typical_rnn_params(
            args, 'simclr_embd_ctl', 
            sy_embd_hipp_funcs.simclr_embd_ctl)


def sc_fast_weights(args):
    return get_typical_rnn_params(
            args, 'sc_fast_weights', 
            sy_embd_hipp_funcs.sc_fast_weights)


def simclr_p200_embd_ctl(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl)


def sc_p200_ctl_layernorm(args):
    return get_typical_rnn_params(
            args, 'sc_p200_ctl_layernorm', 
            sy_embd_hipp_funcs.sc_p200_ctl_layernorm)


def sc_p200_ctl_tanh(args):
    return get_typical_rnn_params(
            args, 'sc_p200_ctl_tanh', 
            sy_embd_hipp_funcs.sc_p200_ctl_tanh)


def sc_p200_ctl_notile(args):
    return get_typical_rnn_params(
            args, 'sc_p200_ctl_notile', 
            sy_embd_hipp_funcs.sc_p200_ctl_notile)


def simclr_p200_embd_ctl_bs512(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl_bs512', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl,
            opt_update_interval=4)


def simclr_p200_embd_ctl_slnwd(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl_slnwd', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl_slnwd)


def sc_p200_bs512_slnwd_clp_tanh(args):
    return get_typical_rnn_params(
            args, 'sc_p200_bs512_slnwd_clp_tanh', 
            sy_embd_hipp_funcs.sc_p200_slnwd_tanh,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_bs512_slnwd_clp_tanh_nt(args):
    return get_typical_rnn_params(
            args, 'sc_p200_bs512_slnwd_clp_tanh_nt', 
            sy_embd_hipp_funcs.sc_p200_slnwd_tanh_notile,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_bs512_slnwd_clp_tanh_nt_sm(args):
    return get_typical_rnn_params(
            args, 'sc_p200_bs512_slnwd_clp_tanh_nt_sm', 
            sy_embd_hipp_funcs.sc_p200_slnwd_tanh_notile_sepmlp,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_sl_nt_sm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_sl_nt_sm_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_sl_nt_sepmlp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_slnwd_nt_sm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_slnwd_nt_sm_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_slnwd_nt_sepmlp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_sl_tanh_nt_sm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_sl_tanh_nt_sm_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_sl_tanh_nt_sepmlp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_seq64_sl_nt_sm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_seq64_sl_nt_sm_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_seq64_sl_nt_sepmlp,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_seq64_slnwd_tanh_nt_sepmlp_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_seq64_slnwd_tanh_nt_sepmlp_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_seq64_slnwd_tanh_nt_sepmlp,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_allrl_sm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_allrl_sm_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_allrl_sm,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_gatestp_all_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_gatestp_all_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_gatestp_all,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})

def sc_seq96_gatestp_all_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq96_gatestp_all_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq96_gatestp_all,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})

def sc_p200_seq96_slnwd_tanh_nt_sepmlp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_seq96_slnwd_tanh_nt_sepmlp', 
            sy_embd_hipp_funcs.sc_p200_seq96_slnwd_tanh_nt_sepmlp,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_notile_sm(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_notile_sm', 
            sy_embd_hipp_funcs.sc_seq64_notile_sm)


def sc_seq64_all_sm_sim_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_bs512_clp_4', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_mltmx_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_mltmx_bs512_clp_2', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_mltmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_mxnm_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_mxnm_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_mxnm,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_sgd_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sgd_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_sgd,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_ftsgd_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_ftsgd_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_ftsgd,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_ftsgd_cos_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_ftsgd_cos_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_ftsgd_cos,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_fftsgd_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_fftsgd_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_fftsgd,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_ftAdam_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_ftAdam_bs512_clp_2', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_ftAdam,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_ctmx_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_ctmx_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_ctmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_ctmx(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_ctmx', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_ctmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_sim_gate_2mlps_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_gate_2mlps_bs512_clp', 
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_2mlps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_ctmx_sephp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_ctmx_sephp_2',
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_ctmx_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_ctmx_sephp_re(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_ctmx_sephp_re_2',
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_ctmx_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_sim_gate_ctmx_sephp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_gate_ctmx_sephp',
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_ctmx_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_sim_gate_ctmx_sephp_re(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_gate_ctmx_sephp_re',
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_ctmx_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_new_sim_gate_ctmx_ps(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_ps_3',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_ps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def set_sim_gate_load(params):
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_seq64_all_sm_sim_gate_ctmx_sephp/epoch_350.pth',
            }
    params['max_epochs'] = 800
    return params


def sc_seq64_all_sm_sim_gate_ctmx_sephp_tst(args):
    params = get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_gate_ctmx_sephp_tst',
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_ctmx_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params = set_sim_gate_load(params)
    return params


def sc_seq64_new_sim_gate_ctmx_psld(args):
    params = get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_psld',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_ps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})
    params = set_sim_gate_load(params)
    return params


def sc_seq64_new_sim_gate_ctmx_rawpsld(args):
    params = get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_rawpsld',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_rawps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})
    params = set_sim_gate_load(params)
    return params


def sc_seq64_new_sim_gate_ctmx_nsps(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_nsps',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_nsps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_new_sim_gate_ctmx_ns2ps(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_ns2ps',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_ns2ps,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_sim_gate_sephp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_sim_gate_sephp_2',
            sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_ctmx_sephp_cr(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_ctmx_sephp_cr',
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_ctmx_sephp_cr,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_sephp(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_sephp',
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq64_all_sm_gate_2mlps_sephp_re(args):
    return get_typical_rnn_params(
            args, 'sc_seq64_all_sm_gate_2mlps_sephp_re',
            sy_embd_hipp_funcs.sc_seq64_all_sm_gate_2mlps_sephp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_seq96_new_sim_gate(args):
    return get_typical_rnn_params(
            args, 'sc_seq96_new_sim_gate',
            sy_embd_hipp_funcs.sc_seq96_new_sim_gate,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_varylen_new_sim_gate_ctmx(args):
    return get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0})


def sc_varylen_new_sim_gate_ctmx_ld(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_ld',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params = set_sim_gate_load(params)
    return params


def sc_varylen_new_sim_gate_ctmx_mstft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_mstft_2',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_mstft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params = set_sim_gate_load(params)
    return params


def sc_varylen_new_sim_gate_ctmx_pairft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pairft_2',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params = set_sim_gate_load(params)
    return params


def sc_varylen_new_sim_gate_ctmx_pairft_u4(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pairft_u4',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params = set_sim_gate_load(params)
    params['max_epochs'] = 1200
    return params


def sc_varylen_new_sim_gate_ctmx_pairft_tst(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pairft_tst',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_pairft_2/epoch_370.pth',
            }
    params['max_epochs'] = 800
    return params


def sc_varylen_new_sim_gate_ctmx_tst(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_tst',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_varylen_new_sim_gate_ctmx/epoch_200.pth',
            }
    params['max_epochs'] = 800
    return params


def sc_varylen_new_sim_gate_ctmx_pairft_nld(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pairft_nld',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_pft_ld270(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pft_ld270_2',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_pairft_nld/epoch_270.pth',
            }
    params['max_epochs'] = 800
    return params


def sc_varylen_new_sim_gate_ctmx_pft_u8_nld(args):
    def _inc_lr(cfg):
        #cfg.optimizer['lr'] = 1e-4
        cfg.optimizer['lr'] = 5e-5
        return cfg
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_pft_u8_nld_2',
            gnrl_funcs.sequential_func(
                sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
                _inc_lr),
            opt_update_interval=8,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_rth_pft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_rth_pft',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rth_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_rth_pft_test128(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_rth_pft_test128',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rth_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(128,))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_rth_pft/epoch_600.pth',
            }
    return params


def sc_pft_vlen64_reluth(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_hlf(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_hlf',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_hlf,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_4th(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_4th',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_4th,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_test128(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_test128',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(128,))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_pft_vlen64_reluth/latest_cached.pth',
            }
    return params


def sc_varylen_new_sim_gate_ctmx_rth_pft_rps(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_rth_pft_rps',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rth_pft_rps,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_resmlp_rth_pft_rps(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_resmlp_rth_pft_rps',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_resmlp_rth_pft_rps,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen_reluth_samps(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen_reluth_samps',
            sy_embd_hipp_funcs.sc_pft_vlen_reluth_samps,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen_reluth_samps_p400(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen_reluth_samps_p400',
            sy_embd_hipp_funcs.sc_pft_vlen_reluth_samps_p400,
            opt_update_interval=8,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_lm(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_lm',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_lm,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_op(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_op',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_op,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3_2',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def  sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3_unl(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3_unl_2',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgca3_unl,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_2',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_fl(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_fl',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_fl,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_fl_fp16(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_fl_fp16',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_fl,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_ffl(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_ffl',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_dgdynca3_ffl,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_step4(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_step4',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3_step4,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_step8(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_step8',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3_step8,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_rep8(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_rep8',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3_rep8,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_rep4(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_rep4',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3_rep4,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64))
    return params


def sc_pft_vlen64_p300n_dgdynca3_s8r16(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3_s8r16',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3_s8r16,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dgdynca3ca1(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3ca1',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3ca1,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params

def sc_pft_p300n_dgdynca3ca1(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_p300n_dgdynca3ca1',
            sy_embd_hipp_funcs.sc_pft_p300n_dgdynca3ca1,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=None,
            opt_use_fp16=True,
            )
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_sy_embd/sc_pft_p300n_dgdynca3ca1/latest_cached.pth',
            'resume_optimizer': False
            }
    return params

def sc_pft_p300n_dgdynca3ca1_seq64(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_p300n_dgdynca3ca1_seq64',
            sy_embd_hipp_funcs.sc_pft_p300n_dgdynca3ca1_seq64,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=None,
            opt_use_fp16=True,
            )
    return params

def sc_pft_p300n_dgdynca3ca1_seq96(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_p300n_dgdynca3ca1_seq96',
            sy_embd_hipp_funcs.sc_pft_p300n_dgdynca3ca1_seq96,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=None,
            opt_use_fp16=True,
            )
    return params

def sc_pft_p300n_aha_dgca3(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_p300n_aha_dgca3_test2',
            sy_embd_hipp_funcs.sc_pft_p300n_aha_dgca3,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=None,
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dgdynca3ca1_osfilter(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3ca1_osfilter',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3ca1_osfilter,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dgdynca3ca1_osfilter(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3ca1_osfilter',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3ca1_osfilter,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dgdynca3ca1_top1mx(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dgdynca3ca1_top1mx',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dgdynca3ca1_top1mx,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_top1mx_smq(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_top1mx_smq',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_top1mx_smq,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_r32_osf_smq',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf_prcmp(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf_prcmp',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf_prcmp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_sg_osf_smq_hlf_prcmp(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_sg_osf_smq_hlf_prcmp',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_sg_osf_smq_hlf_prcmp,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_sg_osf_smq_4th_prcmp(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_sg_osf_smq_4th_prcmp',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_sg_osf_smq_4th_prcmp,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_slw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_slw',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_4th_typ_slw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_sslw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_sslw',
            sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_4th_typ_sslw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_u1(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_u1',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_fw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_fw',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_fw,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_ll(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_ll',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_ll,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_l(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_l',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_l,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_p300sn_l_filter(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_p300sn_l_filter',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_p300sn_l_filter,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_p300sn_l_filter_fst(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_p300sn_l_filter_fst',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_p300sn_l_filter_fst,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_p300sn_l_filter_fst_u1(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_p300sn_l_filter_fst_u1',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_p300sn_l_filter_fst,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_p300sn_l_osfilter(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_p300sn_l_osfilter',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_p300sn_l_osfilter,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_p300sn_l_osfilter_fst(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_p300sn_l_osfilter_fst',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_p300sn_l_osfilter_fst,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_s(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_s',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_s,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300_nsdcy_fxec,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_fxec_rec',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_fxec_rec,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_wc(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_fxec_rec_wc',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_fxec_rec_wc,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_lm(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_fxec_rec_lm',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_fxec_rec_lm,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen64_reluth_samps_p300n_fxec_rec_tr(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_reluth_samps_p300n_fxec_rec_tr',
            sy_embd_hipp_funcs.sc_pft_vlen64_reluth_samps_p300n_fxec_rec_tr,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_pft_vlen_reluth_samps_p400_nsdcy(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen_reluth_samps_p400_nsdcy',
            sy_embd_hipp_funcs.sc_pft_vlen_reluth_samps_p400_nsdcy,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_mx_pft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_mx_pft',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_mx_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_mxl4_pft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_mxl4_pft',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_mxl4_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_mlp2_pft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_mlp2_pft',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_mlp2_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_new_sim_gate_ctmx_rthmx_pft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_new_sim_gate_ctmx_rthmx_pft',
            sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rthmx_pft,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_naive_pairft(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_naive_pairft',
            sy_embd_hipp_funcs.sc_varylen_naive_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def sc_varylen_naive_pairft_ld(args):
    params = get_typical_rnn_params(
            args, 'sc_varylen_naive_pairft_ld',
            sy_embd_hipp_funcs.sc_varylen_naive_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    params['load_params'] = {
            'from_checkpoint': '/mnt/fs4/chengxuz/openselfsup_models/work_dirs/new_pipelines/hipp_sy_embd/sc_p200_seq64_slnwd_tanh_nt_sepmlp_bs512_clp/epoch_160.pth',
            }
    params['max_epochs'] = 800
    return params


def sc_seq64_new_sim_gate_ctmx_pairft(args):
    params = get_typical_rnn_params(
            args, 'sc_seq64_new_sim_gate_ctmx_pairft',
            sy_embd_hipp_funcs.sc_seq64_new_sim_gate_ctmx_pairft,
            opt_update_interval=2,
            opt_grad_clip={'max_norm': 1.0},
            )
    params = set_sim_gate_load(params)
    return params


def sc_pft_varylen_fast_weights(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_varylen_fast_weights',
            sy_embd_hipp_funcs.sc_pft_varylen_fast_weights,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96))
    return params


def simclr_p200_embd_ctl_sl(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl_sl', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl_sl)


def simclr_p200_embd_ctl_ft(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl_ft', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl_ft)


def simclr_p200_embd_ctl_clp(args):
    return get_typical_rnn_params(
            args, 'simclr_p200_embd_ctl_clp_2', 
            sy_embd_hipp_funcs.simclr_p200_embd_ctl,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_gate_2mlps(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gate_2mlps', 
            sy_embd_hipp_funcs.sc_p200_gate_2mlps)


def sc_p200_gate_2mlps_newh_pm(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gate_2mlps_newh_pm', 
            sy_embd_hipp_funcs.sc_p200_gate_2mlps_newh_pm)


def simclr16_embd_ctl(args):
    return get_typical_rnn_params(
            args, 'simclr16_embd_ctl', 
            sy_embd_hipp_funcs.simclr16_embd_ctl)


def simclr_embd_gate_stpmlp(args):
    return get_typical_rnn_params(
            args, 'simclr_embd_gate_stpmlp_2', 
            sy_embd_hipp_funcs.simclr_embd_gate_stpmlp)


def simclr_embd_stpmlp(args):
    return get_typical_rnn_params(
            args, 'simclr_embd_stpmlp_2',
            sy_embd_hipp_funcs.simclr_embd_stpmlp)


def sc_p200_gstpmlp_newh_pm(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gstpmlp_newh_pm', 
            sy_embd_hipp_funcs.sc_p200_gstpmlp_newh_pm)


def sc_p200_gstpmlp_newh_pm_bs512(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gstpmlp_newh_pm_bs512', 
            sy_embd_hipp_funcs.sc_p200_gstpmlp_newh_pm,
            opt_update_interval=4)


def sc_p200_sepmlp_slnwd_tanh_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_sepmlp_slnwd_tanh_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_sepmlp_slnwd_tanh,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_sepmlp_slnwd_nt_tanh_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_sepmlp_slnwd_nt_tanh_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_sepmlp_slnwd_tanh_notile,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_gate_sepmlp_slnwd_nt_tanh_bs512_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gate_sepmlp_slnwd_nt_tanh_bs512_clp', 
            sy_embd_hipp_funcs.sc_p200_gate_sepmlp_slnwd_tanh_nt,
            opt_update_interval=4,
            opt_grad_clip={'max_norm': 1.0})


def sc_p200_gstpmlp_newh_pm_nwd(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gstpmlp_newh_pm_nwd', 
            sy_embd_hipp_funcs.sc_p200_gstpmlp_newh_pm_nwd)


def sc_p200_gstpmlp_newh_pm_sl_nwd_clp(args):
    return get_typical_rnn_params(
            args, 'sc_p200_gstpmlp_newh_pm_sl_nwd_clp', 
            sy_embd_hipp_funcs.sc_p200_gstpmlp_newh_pm_sl_nwd,
            opt_grad_clip={'max_norm': 1.0})


def moco_embd_ctl(args):
    return get_typical_rnn_params(
            args, 'moco_embd_ctl', 
            sy_embd_hipp_funcs.moco_embd_ctl)


def moco16_embd_ctl(args):
    return get_typical_rnn_params(
            args, 'moco16_embd_ctl', 
            sy_embd_hipp_funcs.moco16_embd_ctl)
