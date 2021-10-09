import os
import openselfsup
import openselfsup.config_related.alxnt as alxnt
import openselfsup.config_related.vae_funcs as vae_funcs
import openselfsup.config_related.sy_embd_hipp_funcs as sy_embd_hipp_funcs
import openselfsup.config_related.sy_embd_pat_sep as sy_embd_pat_sep
import openselfsup.config_related.imgnt_cont_funcs as imgnt_cont_funcs

import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.ewc_cfg_funcs as ewc_cfg_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import openselfsup.config_related.mst_funcs as mst_funcs


FRMWK_REPO_PATH = os.path.dirname(openselfsup.__path__[0])
MODEL_CKPT_DIR = '/home/an633/project/CuriousContrast/results_alex/'

TYPICAL_SIMCLR_R18_PATH = os.path.join(
  
        FRMWK_REPO_PATH, 'configs/selfsup/simclr/r18.py')
MODEL_KWARGS = dict(
        simclr_alxnt_ctl64=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=alxnt.ctl64,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'simclr_alxnt/ctl64/latest_cached.pth')),
        simclr_alxnt_sy_two_img_ctl64=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=alxnt.sy_two_img_ctl64,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 
                'simclr_alxnt/sy_two_img_ctl64/latest_cached.pth')),
        simclr_alxnt_sy_two_img_rd_ctl64=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=alxnt.sy_two_img_rd_ctl64,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 
                'simclr_alxnt/sy_two_img_rd_ctl64/latest_cached.pth')),
        simclr_sy_ctl=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, 
            cfg_func=lambda cfg: cfg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 
                'simclr/r18_sy_ctl/latest_cached.pth')),
        simclr_in_ctl=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=lambda cfg: cfg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_ep300/latest_cached.pth')),
        simclr_in_bld=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=imgnt_cont_funcs.imgnt_batchLD_cfg_func,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_in_bld_ep300/latest_cached.pth')),
        simclr_in_ctl_bld=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=imgnt_cont_funcs.imgnt_batchLD_cfg_func,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_ep300/latest_cached.pth')),
        simclr_mst_ft_in=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=lambda cfg: cfg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr_mstft/r18_in10_fx/epoch_302.pth')),
        simclr_mst_pair_ft_in=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=lambda cfg: cfg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr_mstft/r18_pair_in/epoch_302.pth')),
        simclr_sy_ep300_ctl=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, 
            cfg_func=saycam_funcs.random_saycam_1M_cfg_func,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 
                'simclr/r18_sy_ctl_ep300/latest_cached.pth')),
        )
COTR_MODEL_KWARGS = dict(
        simclr_mlp4_sy_hctr_is112=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func),
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_sy_hcatctr_is112_mlp4_ep300/latest_cached.pth')),
        simclr_mlp4_sy_hctr_max6k_is112=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max6000_cnd_storage_cfg_func),
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_sy_hcatctr_is112_mlp4_ep300/latest_cached.pth')),
        simclr_mlp4_sy_hctr_max40k_is112=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, 
            cfg_func=gnrl_funcs.sequential_func(
                saycam_funcs.cotrain_saycam_half_ep300_cfg_func,
                gnrl_funcs.res112,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                saycam_funcs.max40k_cnd_storage_cfg_func),
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'simclr/r18_sy_hcatctr_is112_mlp4_ep300/latest_cached.pth')),
        )
VAE_MODEL_KWARGS = dict(
        default_vae=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.ctl_saycam_vae,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'vae/default_vae_2/latest_cached.pth')),
        default_vae64=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.ctl_saycam_vae64,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'vae/default_vae64_2/latest_cached.pth')),
        default_vae112=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.ctl_saycam_vae112,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'vae/default_vae112/latest_cached.pth')),
        inter_vae_c2_lessZ=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.saycam_inter_vae_c2_lessZ,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'vae/saycam_inter_vae_c2_lessZ/latest_cached.pth')),
        interbn_vae_c2=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.saycam_interbn_vae_c2,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR,
                'vae/saycam_interbn_vae_c2/latest_cached.pth')),
        interbn_vae_c2_w_vae112=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH, cfg_func=vae_funcs.test_saycam_interbn_vae_c2_w_vae112,
            ckpt_path=None),
        )
HIPP_MODEL_KWARGS = dict(
        simple_gate_early=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_ctmx_sephp,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_seq64_all_sm_sim_gate_ctmx_sephp/epoch_100.pth')),
        simple_gate=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_seq64_all_sm_sim_gate_ctmx_sephp,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_seq64_all_sm_sim_gate_ctmx_sephp/epoch_350.pth')),
        simple_gate_varylen=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_varylen_new_sim_gate_ctmx/epoch_200.pth')),
        simple_gate_ft_vlen_rth=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rth_pft,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_rth_pft/epoch_530.pth')),
        simple_gate_ft_vlen_rth_e=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_rth_pft,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_rth_pft/epoch_200.pth')),
        simple_gate_ft_vlen=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_varylen_new_sim_gate_ctmx_pairft,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_varylen_new_sim_gate_ctmx_pairft_nld/epoch_270.pth')),
        dynca1_hlf=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_pft_vlen64_p300n_dynca1_r32_osf_smq_hlf/latest_cached.pth')),
        dynca1_4th_prcmp=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_pft_vlen64_p300n_dynca1_r32_osf_smq_4th_prcmp/latest_cached.pth')),
        dynca1_4th_sg_sslw=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_hipp_funcs.sc_pft_vlen64_p300n_dynca1_4th_typ_sslw,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_sy_embd/sc_pft_vlen64_p300n_dynca1_4th_typ_sslw/latest_cached.pth')),
        simple_gate_mst=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=mst_funcs.mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'hipp_simclr/mst_saycam_mlpreadout_patrep_cfg_func_nonlinearneckv3/latest_cached.pth')),
        )
PAT_SEP_MODEL_KWARGS = dict(
        naive_mlp_dg=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_dg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_dg_2/latest_cached.pth')),
        naive_mlp_dw_dg=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_dw_dg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_dw_dg_2/latest_cached.pth')),
        naive_mlp_ww_dg=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_ww_dg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_ww_dg_2/latest_cached.pth')),
        naive_mlp_s30_ww_dg=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_s30_ww_dg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_s30_ww_dg/latest_cached.pth')),
        naive_mlp_s40_ww_dg=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_s40_ww_dg,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_s40_ww_dg/latest_cached.pth')),
        naive_mlp_dg_rec=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_dg_rec,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_dg_rec/epoch_40.pth')),
        naive_mlp_s30_ww_dg_rec=dict(
            cfg_path=TYPICAL_SIMCLR_R18_PATH,
            cfg_func=sy_embd_pat_sep.naive_mlp_s30_ww_dg_rec,
            ckpt_path=os.path.join(
                MODEL_CKPT_DIR, 'sy_embd_pat_sep/naive_mlp_s30_ww_dg_rec_2/epoch_30.pth')),
        )
