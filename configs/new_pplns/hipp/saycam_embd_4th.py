from .saycam_embd import *
import openselfsup.config_related.sy_embd_hipp_funcs_4th as sy_embd_hipp_funcs_4th


def sc_pft_vlen64_p300n_dynca1_4th_typ_ssslw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_ssslw',
            sy_embd_hipp_funcs_4th.sc_pft_vlen64_p300n_dynca1_4th_typ_ssslw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_sslw_v5(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_sslw_v5',
            sy_embd_hipp_funcs_4th.sc_pft_vlen64_p300n_dynca1_4th_typ_sslw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            valid_interval=5,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_slw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_slw_3',
            sy_embd_hipp_funcs_4th.sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_slw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            valid_interval=5,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_sslw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_sslw_3',
            sy_embd_hipp_funcs_4th.sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_sslw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            valid_interval=5,
            )
    return params


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_ssslw(args):
    params = get_typical_rnn_params(
            args, 'sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_ssslw_2',
            sy_embd_hipp_funcs_4th.sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_ssslw,
            opt_update_interval=1,
            opt_grad_clip={'max_norm': 1.0},
            vary_len_val=(32, 64, 96),
            opt_use_fp16=True,
            valid_interval=5,
            )
    return params
