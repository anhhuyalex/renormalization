from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.hipp_funcs as hipp_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_rnn_params(args, exp_id, cfg_func):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = ParamsBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='hipp_simclr',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    params['validation_params'] = {}
    return params


def rnn_test(args):
    return get_typical_rnn_params(
            args, 'rnn_test', hipp_funcs.hipp_rnn_test_cfg_func)


def rnn_adam(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_3', hipp_funcs.hipp_rnn_adam_cfg_func)


def rnn_adam_self(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_self_2', hipp_funcs.hipp_rnn_adam_self_cfg_func)


def rnn_adam_self_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_self_jp', 
            hipp_funcs.hipp_rnn_adam_self_jp_cfg_func)


def rnn_adam_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_pat_jp_7', 
            hipp_funcs.hipp_rnn_adam_pat_jp_cfg_func)


def rnn_adam_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_pat', 
            hipp_funcs.hipp_rnn_adam_pat_cfg_func)


def rnn_adam_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_gnrl_pat_jp', 
            hipp_funcs.hipp_rnn_adam_gnrl_pat_jp_cfg_func)


def rnn_adam_mh_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mh_gnrl_pat_jp', 
            hipp_funcs.hipp_rnn_adam_mh_gnrl_pat_jp_cfg_func)


def rnn_adam_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mmax_gnrl_pat_jp_4', 
            hipp_funcs.hipp_rnn_adam_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_mh_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mh_mmax_gnrl_pat_jp_3', 
            hipp_funcs.hipp_rnn_adam_mh_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_mh_mmax_gnrl_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mh_mmax_gnrl_pat', 
            hipp_funcs.hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func)


def rnn_adam_mh_pr_mmax_gnrl_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mh_pr_mmax_gnrl_pat', 
            hipp_funcs.hipp_rnn_adam_mh_pr_mmax_gnrl_pat_cfg_func)


def rnn_adam_at_mh_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_at_mh_mmax_gnrl_pat_jp', 
            hipp_funcs.hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_atD_mh_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atD_mh_mmax_gnrl_pat_jp', 
            hipp_funcs.hipp_rnn_adam_atD_mh_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_atDD_mh_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atDD_mh_mmax_gnrl_pat_jp_2', 
            hipp_funcs.hipp_rnn_adam_atDD_mh_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_atDDH_mh_mmax_gnrl_pat_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atDDH_mh_mmax_gnrl_pat_jp', 
            hipp_funcs.hipp_rnn_adam_atDDH_mh_mmax_gnrl_pat_jp_cfg_func)


def rnn_adam_atDDH_lng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atDDH_lng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_atDDH_lng_mlpmax_jp_cfg_func)


def rnn_adam_atLDDHH_lng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atLDDHH_lng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_atLDDHH_lng_mlpmax_jp_cfg_func)


def rnn_adam_atDDH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atDDH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_atDDH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_atDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_stpmlp_atDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_stpmlp_atDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_stpmlp_atDDHH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_seqctl_atDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_seqctl_atDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_seqctl_atDDHH_lnglng_mlpmax_jp_cfg_func)


def seqctl_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'seqctl_gate_hsim32_jp', 
            hipp_funcs.hipp_seqctl_gate_hsim32_jp_cfg_func)


def seqctl_cntx_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'seqctl_cntx_hsim32_jp', 
            hipp_funcs.hipp_seqctl_cntx_hsim32_jp_cfg_func)


def seqctl200_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'seqctl200_gate_hsim32_jp', 
            hipp_funcs.hipp_seqctl200_gate_hsim32_jp_cfg_func)


def rc_seqctl_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'rc_seqctl_gate_hsim32_jp', 
            hipp_funcs.hipp_rc_seqctl_gate_hsim32_jp_cfg_func)


def stpmlp_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'stpmlp_gate_hsim32_jp', 
            hipp_funcs.hipp_stpmlp_gate_hsim32_jp_cfg_func)


def lstm_stpmlp_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'lstm_stpmlp_gate_hsim32_jp', 
            hipp_funcs.hipp_lstm_stpmlp_gate_hsim32_jp_cfg_func)


def gate_stpmlp_gate_hsim32_jp(args):
    return get_typical_rnn_params(
            args, 'gate_stpmlp_gate_hsim32_jp', 
            hipp_funcs.hipp_gate_stpmlp_gate_hsim32_jp_cfg_func)


def rnn_adam_atLDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atLDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func)

def rnn_adam_atLDDHH_lnglng_mlpmax_jp_heterogeneous(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_atLDDHH_lnglng_mlpmax_jp_heterogeneous', 
            hipp_funcs.hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func_heterogeneous)

def hipp_rnn_fastweights_heterogeneous(args):
    return get_typical_rnn_params(
            args, 'hipp_rnn_fastweights_heterogeneous', 
            hipp_funcs.hipp_rnn_fastweights_heterogeneous)

def hipp_rnn_adam_hopfield_heterogeneous(args):
    return get_typical_rnn_params(
            args, 'hipp_rnn_adam_hopfield_heterogeneous', 
            hipp_funcs.hipp_rnn_adam_hopfield_heterogeneous)


def rnn_adam_stpmlp_atLDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_stpmlp_atLDDHH_lnglng_mlpmax_jp_2', 
            hipp_funcs.hipp_rnn_adam_stpmlp_atLDDHH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_2simstpmlp_atLDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_2simstpmlp_atLDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_2simstpmlp_atLDDHH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_seqctl_atLDDHH_lnglng_mlpmax_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_seqctl_atLDDHH_lnglng_mlpmax_jp', 
            hipp_funcs.hipp_rnn_adam_seqctl_atLDDHH_lnglng_mlpmax_jp_cfg_func)


def rnn_adam_at_mh_pr_mmax_gnrl_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_at_mh_pr_mmax_gnrl_pat', 
            hipp_funcs.hipp_rnn_adam_at_mh_pr_mmax_gnrl_pat_cfg_func)


def rnn_adam_mh_pr_nh_mmax_gnrl_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mh_pr_nh_mmax_gnrl_pat', 
            hipp_funcs.hipp_rnn_adam_mh_pr_nh_mmax_gnrl_pat_cfg_func)


def rnn_adam_at_mh_pr_nh_mmax_gnrl_pat(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_at_mh_pr_nh_mmax_gnrl_pat', 
            hipp_funcs.hipp_rnn_adam_at_mh_pr_nh_mmax_gnrl_pat_cfg_func)


def rnn_adam_dw(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dw_2', hipp_funcs.hipp_rnn_adam_dw_cfg_func)


def rnn_adam_dw_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dw_jp', 
            hipp_funcs.hipp_rnn_adam_dw_jp_cfg_func)


def rnn_adam_dw_l2(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dw_l2', hipp_funcs.hipp_rnn_adam_dw_l2_cfg_func)


def rnn_adam_simple_rnn_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_simple_rnn_jp', 
            hipp_funcs.hipp_rnn_adam_simple_rnn_jp_cfg_func)


def rnn_adam_dw_simple_rnn_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dw_simple_rnn_jp', 
            hipp_funcs.hipp_rnn_adam_dw_simple_rnn_jp_cfg_func)


def rnn_adam_dwt_simple_rnn_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dwt_simple_rnn_jp', 
            hipp_funcs.hipp_rnn_adam_dwt_simple_rnn_jp_cfg_func)


def rnn_adam_wdr(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_wdr_3', hipp_funcs.hipp_rnn_adam_wdr_cfg_func)


def rnn_adam_wdr_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_wdr_jp', hipp_funcs.hipp_rnn_adam_wdr_jp_cfg_func)


def rnn_adam_dwt_wdr_jp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_dwt_wdr_jp', 
            hipp_funcs.hipp_rnn_adam_dwt_wdr_jp_cfg_func)


def rnn_adam_id(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id', hipp_funcs.hipp_rnn_adam_id_cfg_func)


def rnn_adam_id_l2(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_l2', hipp_funcs.hipp_rnn_adam_id_l2_cfg_func)


def rnn_adam_id_l4(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_l4', hipp_funcs.hipp_rnn_adam_id_l4_cfg_func)


def rnn_adam_l4(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_l4', hipp_funcs.hipp_rnn_adam_l4_cfg_func)


def rnn_adam_id_n0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_n0', hipp_funcs.hipp_rnn_adam_id_n0_cfg_func)


def rnn_adam_id_relu(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_relu_4', 
            hipp_funcs.hipp_rnn_adam_id_relu_cfg_func)


def rnn_adam_id_relu_l2(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_relu_l2', 
            hipp_funcs.hipp_rnn_adam_id_relu_l2_cfg_func)


def rnn_adam_id_t3(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_id_t3_2', 
            hipp_funcs.hipp_rnn_adam_id_t3_cfg_func)


def rnn_adam_hp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_2', 
            hipp_funcs.hipp_rnn_adam_hp_cfg_func)


def rnn_adam_hp_u0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_u0', 
            hipp_funcs.hipp_rnn_adam_hp_u0_cfg_func)


def rnn_adam_hp_st3_u0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_st3_u0', 
            hipp_funcs.hipp_rnn_adam_hp_st3_u0_cfg_func)


def rnn_adam_hp_u0_n0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_u0_n0', 
            hipp_funcs.hipp_rnn_adam_hp_u0_n0_cfg_func)


def rnn_adam_hp_pr4_u0_n0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_pr4_u0_n0', 
            hipp_funcs.hipp_rnn_adam_hp_pr4_u0_n0_cfg_func)


def rnn_adam_hp_pr4_s3_u0_n0(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_hp_pr4_s3_u0_n0', 
            hipp_funcs.hipp_rnn_adam_hp_pr4_s3_u0_n0_cfg_func)


def rnn_adam_mlp(args):
    return get_typical_rnn_params(
            args, 'rnn_adam_mlp', 
            hipp_funcs.hipp_rnn_adam_mlp_cfg_func)


def adam_mlp(args):
    return get_typical_rnn_params(
            args, 'adam_mlp_2', 
            hipp_funcs.hipp_adam_mlp_cfg_func)

def classicalhopfield(args):
    return get_typical_rnn_params(
            args, 'classicalhopfield', 
            hipp_funcs.hipp_adam_classicalhopfield_alex)

def classicalhopfield_correlated_vecs(args):
    return get_typical_rnn_params(
            args, 'classicalhopfield_correlated_vecs', 
            hipp_funcs.hipp_adam_classicalhopfield_alex_correlated_vecs)
