import torch
import torch.nn as nn
import torch.optim as optim

import utils

def get_configs(args):
    # Get model optim params
    if args.model_name == "mlp":
        net = utils.MLP(input_size=(32*32*3), hidden_sizes=[100,100,100],
                   activation = "relu").cuda()
        
    elif args.model_name == "cnn":
        net = utils.CNN().cuda()
        
    elif args.model_name == "cnn_chan_1-16":
        net = utils.CNN(conv1_chans = 1, conv2_chans = 16).cuda()
    elif args.model_name == "cnn_chan_1-1":
        net = utils.CNN(conv1_chans = 1, conv2_chans = 1).cuda()
        
    elif args.model_name == "attn":
        import attention
        net = attention.SimpleViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 1024,
            depth = 1,
            heads = 16,
            mlp_dim = 2048
        ).cuda()
        
    elif args.model_name == "attn_no_pe":
        import attention
        net = attention.SimpleViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 1024,
            depth = 1,
            heads = 16,
            mlp_dim = 2048,
            pe_weight = 0
        ).cuda()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer)
    
    data_params = dict(pixel_shuffled = args.pixel_shuffled)
    
    train_params = dict(batch_size = 4, 
                        num_epochs = 100)
    
    
    save_params = dict(save_dir = args.save_dir)
    
    cfg = dict(
        model_optim_params = model_optim_params,
        data_params = data_params,
        train_params = train_params,
        save_params = save_params
    ) 
    return cfg