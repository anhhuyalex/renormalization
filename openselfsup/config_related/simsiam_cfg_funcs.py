def more_hid_bn(cfg):
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=512,
            num_layers=3,
            sync_bn=True,
            with_bias=True,
            with_last_bn=True,
            with_avg_pool=True)
    cfg.model['head']['predictor']['hid_channels'] = 512
    return cfg
