def sequential_func(*args):
    def ret_func(cfg):
        for _func in args:
            cfg = _func(cfg)
        return cfg
    return ret_func


def res112(cfg):
    def _change_one(data_cfg):
        if 'pipeline' in data_cfg:
            data_cfg['pipeline'][0]['size'] = 112
        elif 'pipeline1' in data_cfg:
            data_cfg['pipeline1'][0]['size'] = 112
            data_cfg['pipeline2'][0]['size'] = 112
        else:
            raise NotImplementedError
        return data_cfg
    cfg.data['train'] = _change_one(cfg.data['train'])
    if 'train1' in cfg.data:
        cfg.data['train1'] = _change_one(cfg.data['train1'])
    if 'train2' in cfg.data:
        cfg.data['train2'] = _change_one(cfg.data['train2'])
    return cfg
