def more_mlp_layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 3
    cfg.model['head']['predictor']['num_layers'] = 3
    return cfg


def mlp_4layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['head']['predictor']['num_layers'] = 4
    return cfg


def mlp_4L1bn_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['neck']['bn_settings'] = [True, False, False]
    cfg.model['head']['predictor']['num_layers'] = 4
    cfg.model['head']['predictor']['bn_settings'] = [True, False, False]
    return cfg


def mlp_5layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 5
    cfg.model['head']['predictor']['num_layers'] = 5
    return cfg


def mlp_6layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 6
    cfg.model['head']['predictor']['num_layers'] = 6
    return cfg
