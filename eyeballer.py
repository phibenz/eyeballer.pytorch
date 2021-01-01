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

LABELS = ["Homepage", "Login Page", "Not Found", "Old Looking"]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Eyeballer')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Folder containing the images to test')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'vgg19','resnet18', 'resnet50', 'resnet101', 'resnet152'],
                        help='Model architecture: (default: vgg16)')
    parser.add_argument('--seed', type=int, default=111,
                        help='Seed used (default: 111)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()

    return args

def main():
    args = parse_arguments()

    random.seed(args.seed)
    cudnn.benchmark = True

    model_path = get_model_path('websites', args.arch, args.seed)

    # Data specifications for the webistes dataset
    mean = [0., 0., 0.]
    std = [1., 1., 1.]
    input_size = 224
    num_classes = 4

    # Dataset
    test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    data_test = dset.ImageFolder(root=args.test_dir, transform=test_transform)

    # Dataloader
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
    
    # Loading the checkpoint
    net.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pth.tar'))['state_dict'])
    net.eval()

    # Cuda
    if args.use_cuda:
        net.cuda()

    for idx, (img, _) in enumerate(data_test_loader):
        if args.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            pred = torch.argmax(net(img), dim=-1)
        
        samples = data_test.samples[idx*args.batch_size:(idx+1)*args.batch_size]
        for idx2, sample in enumerate(samples):
            label_idx = pred[idx2].cpu().detach().numpy()
            label = LABELS[label_idx]
            print("{} - {} - {}".format(sample[0], label , label_idx))
    
if __name__ == '__main__':
    main()
