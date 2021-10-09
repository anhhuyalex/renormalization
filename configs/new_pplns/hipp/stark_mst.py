from ..basic_param_setter import ParamsBuilder
from ..mst_param_setter import MSTParamBuilder
import openselfsup.config_related.mst_funcs as mst_funcs
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

def mst_synth_cfg_func(args):
    return get_typical_rnn_params(
            args, 'mst_synth_cfg_func', mst_funcs.mst_synth_cfg_func)

def mst_saycam_cfg_func(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_cfg_func', mst_funcs.mst_saycam_cfg_func)
    return params

def mst_saycam_readout_cfg_func(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_readout_cfg_func', mst_funcs.mst_saycam_readout_cfg_func)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_readout_cfg_func_350/simple_gate_model_ep100.pth',
            'resume_optimizer': False
            }
    return params


def mst_saycam_mlpreadout_patrep_cfg_func(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_patrep_cfg_func', mst_funcs.mst_saycam_mlpreadout_patrep_cfg_func)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_readout_cfg_func_350/simple_gate_model_ep350.pth',
            'resume_optimizer': False
            }
    return params

def mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3', mst_funcs.mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_readout_cfg_func_350/simple_gate_model_ep350.pth',
            'resume_optimizer': False
            }
    return params

def mst_saycam_mlpreadout_rawpatrep_cfg_func(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_rawpatrep_cfg_func', mst_funcs.mst_saycam_mlpreadout_rawpatrep_cfg_func)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_readout_cfg_func_350/simple_gate_model_ep350.pth',
            'resume_optimizer': False
            }
    return params

def mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3', mst_funcs.mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_readout_cfg_func_350/simple_gate_model_ep350.pth',
            'resume_optimizer': False
            }
    return params

def mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion', mst_funcs.mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3/latest_cached.pth',
            'resume_optimizer': False
            }
    return params

def mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion(args):
    params = get_typical_rnn_params(
            args, 'mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion', mst_funcs.mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3_confusion)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_mlpreadout_rawpatrep_cfg_func_nonlinearneckv3/latest_cached.pth',
            'resume_optimizer': False
            }
    return params

def get_mst_rnn_params(args, exp_id, cfg_func):
    cfg_path = BASIC_SIMCLR_CFG
    param_builder = MSTParamBuilder(
            args=args, exp_id=exp_id,
            cfg_path=cfg_path, 
            add_svm_val=True, col_name='hipp_simclr',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params

def mst_saycam_val_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion(args):
    params = get_mst_rnn_params(
            args, 'mst_saycam_val_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion', mst_funcs.mst_saycam_val_mlpreadout_patrep_cfg_func_nonlinearneckv3_confusion)
    params['load_params'] = {
            'resume': True,
            'from_checkpoint': '/home/an633/project/CuriousContrast/results_alex/hipp_simclr/mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3/latest_cached.pth',
            'resume_optimizer': False
            }
    return params