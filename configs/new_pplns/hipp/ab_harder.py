from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.harder_hipp_funcs as harder_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_rnn_params(args, exp_id, cfg_func):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = ParamsBuilder(
            args, exp_id, cfg_path, 
            add_svm_val=True, col_name='hipp_harder',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    params['validation_params'] = {}
    return params


def msim32_mns(args):
    return get_typical_rnn_params(
            args, 'msim32_mns', 
            harder_funcs.msim32_mns)


def msim32_mns_s(args):
    return get_typical_rnn_params(
            args, 'msim32_mns_s', 
            harder_funcs.msim32_mns_s)


def msim32_mns_s_mask(args):
    return get_typical_rnn_params(
            args, 'msim32_mns_s_mask', 
            harder_funcs.msim32_mns_s_mask)


def msim32_mns_s_mask_stpmlp(args):
    return get_typical_rnn_params(
            args, 'msim32_mns_s_mask_stpmlp', 
            harder_funcs.msim32_mns_s_mask_stpmlp)


def msim32_mns_s_mask_gate_stpmlp(args):
    return get_typical_rnn_params(
            args, 'msim32_mns_s_mask_gate_stpmlp', 
            harder_funcs.msim32_mns_s_mask_gate_stpmlp)
