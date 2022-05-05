import utils
import datetime

def get_configs(args):
    # Get model optim params
    if args.model_name == "mlp":
        net = utils.MLP(input_size=(32*32*3), hidden_sizes=[100,100,100],
                   activation = "relu").cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif args.model_name == "cnn":
        net = utils.CNN().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    model_optim_params = dict(model = net, criterion = criterion, optimizer = optimizer)
    
    data_params = dict(pixel_shuffled = args.pixel_shuffled)
    
    train_params = dict(batch_size = 4, 
                        num_epochs = 20)
    
    exp_name = f"{args.model_name}_" \
                + f"shuffled_{args.pixel_shuffled}" \
                + f"rep_{datetime.datetime.now().timestamp()}"
    save_params = dict(save_dir = args.save_dir, exp_name = exp_name)
    
    cfg = dict(
        model_optim_params = model_optim_params,
        data_params = data_params,
        train_params = train_params,
        save_params = save_params
    ) 
    return cfg