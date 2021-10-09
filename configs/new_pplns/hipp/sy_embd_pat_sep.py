from ..saycam_embd_param_setter import SAYCamEmbdParamBuilder
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.sy_embd_pat_sep as sy_embd_ps_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_rnn_params(
        args, exp_id, cfg_func, vary_len_val=None, **kwargs):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = SAYCamEmbdParamBuilder(
            args=args, exp_id=exp_id, 
            cfg_path=cfg_path,
            vary_len_val=vary_len_val,
            add_svm_val=False, col_name='sy_embd_pat_sep',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            **kwargs)
    params = param_builder.build_params()
    params.pop('extra_hook_params')
    return params


def naive_mlp_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_dg_2', 
            sy_embd_ps_funcs.naive_mlp_dg)


def naive_mlp_dw_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_dw_dg_2', 
            sy_embd_ps_funcs.naive_mlp_dw_dg)


def naive_mlp_ww_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_ww_dg_2', 
            sy_embd_ps_funcs.naive_mlp_ww_dg)


def naive_mlp_s10_ww_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s10_ww_dg', 
            sy_embd_ps_funcs.naive_mlp_s10_ww_dg)


def naive_mlp_s20_ww_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s20_ww_dg', 
            sy_embd_ps_funcs.naive_mlp_s20_ww_dg)


def naive_mlp_s30_ww_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s30_ww_dg', 
            sy_embd_ps_funcs.naive_mlp_s30_ww_dg)


def naive_mlp_s40_ww_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s40_ww_dg', 
            sy_embd_ps_funcs.naive_mlp_s40_ww_dg)


def naive_mlp_s40_dw_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s40_dw_dg', 
            sy_embd_ps_funcs.naive_mlp_s40_dw_dg)


def naive_mlp_s30_dw_dg(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s30_dw_dg', 
            sy_embd_ps_funcs.naive_mlp_s30_dw_dg)


def naive_mlp_dg_rec(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_dg_rec', 
            sy_embd_ps_funcs.naive_mlp_dg_rec)


def naive_mlp_s30_ww_dg_rec(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s30_ww_dg_rec_2', 
            sy_embd_ps_funcs.naive_mlp_s30_ww_dg_rec)


def naive_mlp_s40_ww_dg_rec(args):
    return get_typical_rnn_params(
            args, 'naive_mlp_s40_ww_dg_rec', 
            sy_embd_ps_funcs.naive_mlp_s40_ww_dg_rec)


def h512_mlp_s30_ww_dg_rec(args):
    return get_typical_rnn_params(
            args, 'h512_mlp_s30_ww_dg_rec', 
            sy_embd_ps_funcs.h512_mlp_s30_ww_dg_rec)


def dgca3_pat_sep_rec(args):
    return get_typical_rnn_params(
            args, 'dgca3_pat_sep_rec', 
            sy_embd_ps_funcs.dgca3_pat_sep_rec)
