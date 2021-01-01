import os, sys, random, copy, time
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import torchvision.datasets as dset

from utils.utils import get_model_path, print_log, manipulate_net_architecture
from utils.utils import convert_secs2time, time_string
from utils.training import adjust_learning_rate, train_model, validate, save_checkpoint
from utils.training import RecorderMeter, AverageMeter

from config.config import WEBSITES_DATASET_PATH

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Network')
    # Data and Model options
    parser.add_argument('--dataset', default='websites', choices=['websites'],
                        help='Trainig dataset (default: websites)')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'vgg19','resnet18', 'resnet50', 'resnet101', 'resnet152'],
                        help='Model architecture: (default: vgg16)')
    parser.add_argument('--seed', type=int, default=111,
                        help='Seed used (default: 111)')
    # Optimization options
    parser.add_argument('--loss-function', default='ce', choices=['ce'],
                        help='Loss function (default: ce)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train (dfault: 30)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[],
                        help='Decrease learning rate at these epochs (default: [])')
    parser.add_argument('--gammas', type=float, nargs='+', default=[],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule (default: [])')
    parser.add_argument('--print-freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()

    return args

def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    model_path = get_model_path(args.dataset, args.arch, args.seed)

    # Init logger
    log_file_name = os.path.join(model_path, 'log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('model path : {}'.format(model_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Data specifications for the webistes dataset
    mean = [0., 0., 0.]
    std = [1., 1., 1.]
    input_size = 224
    num_classes = 4

    # Dataset
    traindir = os.path.join(WEBSITES_DATASET_PATH, 'train')
    valdir = os.path.join(WEBSITES_DATASET_PATH, 'val')

    train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    data_train = dset.ImageFolder(root=traindir, transform=train_transform)
    data_test = dset.ImageFolder(root=valdir, transform=test_transform)    

    # Dataloader    
    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    # Network
    if args.arch == "vgg16":
        net = models.vgg16(pretrained=True)
    elif args.arch == "vgg19":
        net = models.vgg19(pretrained=True)
    elif args.arch == "resnet18":
        net = models.resnet18(pretrained=True)
    elif args.arch == "resnet50":
        net = models.resnet50(pretrained=True)
    elif args.arch == "resnet101":
        net = models.resnet101(pretrained=True)
    elif args.arch == "resnet152":
        net = models.resnet152(pretrained=True)
    else:
        raise ValueError("Network {} not supported".format(args.arch))

    if num_classes != 1000:
        net = manipulate_net_architecture(model_arch=args.arch, net=net, num_classes=num_classes)

    # Loss function
    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError
    
    # Cuda
    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    # Optimizer
    momentum = 0.9
    decay = 5e-4
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=momentum, weight_decay=decay, nesterov=True)

    recorder = RecorderMeter(args.epochs)
    start_time = time.time()
    epoch_time = AverageMeter()

    # Main loop
    for epoch in range(args.epochs):
        current_learning_rate = adjust_learning_rate(args.learning_rate, momentum, optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train_model(data_loader=data_train_loader, model=net, criterion=criterion, optimizer=optimizer, epoch=epoch, log=log,
                                            print_freq=200, use_cuda=True)

        # evaluate on test set
        print_log("Validation on test dataset:", log)
        val_acc, val_loss = validate(data_test_loader, net, criterion, log=log, use_cuda=args.use_cuda)
        recorder.update(epoch, train_los, train_acc, val_loss, val_acc)
        
        save_checkpoint({
          'epoch'       : epoch + 1,
          'arch'        : args.arch,
          'state_dict'  : net.state_dict(),
          'optimizer'   : optimizer.state_dict(),
          'args'        : copy.deepcopy(args),
        }, model_path, 'checkpoint.pth.tar')
        
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(model_path, 'curve.png') )

    log.close()

if __name__ == '__main__':
    main()
