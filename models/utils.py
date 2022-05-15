from models.wrn import WideResNet
import torch
from torchvision.models import densenet121
import numpy as np

def build_model(model_type, num_classes, device, args):
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net