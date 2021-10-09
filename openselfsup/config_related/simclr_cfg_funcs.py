def lower_tau(cfg):
    cfg.model['head']['temperature'] = 0.07
    return cfg


def neg_fn_num(cfg):
    cfg.model['head']['neg_fn_num'] = 400
    return cfg


def mneg_fn_num(cfg):
    cfg.model['head']['neg_fn_num'] = 470
    return cfg


def mlp_3layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 3
    return cfg


def mlp_4layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    return cfg


def mlp_5layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 5
    return cfg


def mlp_4L1bn_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['neck']['bn_settings'] = [True, False, False]
    return cfg


def neg_th_value_d7(cfg):
    cfg.model['head']['neg_th_value'] = 0.7
    return cfg


def neg_th_value_d5(cfg):
    cfg.model['head']['neg_th_value'] = 0.5
    return cfg


def neg_th_value_d9(cfg):
    cfg.model['head']['neg_th_value'] = 0.9
    return cfg
