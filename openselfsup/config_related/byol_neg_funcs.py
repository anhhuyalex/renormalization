def byol_neg_cfg_func(cfg):
    #cfg.model['predictor'] = cfg.model['head']['predictor']
    cfg.model['predictor'] = dict(type='Identity')
    cfg.model['head'] = dict(type='ContrastiveHead', temperature=0.1)
    cfg.model['type'] = 'BYOLNeg'
    return cfg
