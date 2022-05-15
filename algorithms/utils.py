

def create_alg(args, device, num_classes, train_loader):
    if args.alg == "standard":
        from algorithms.standard import Standard
        alg_obj = Standard(args, device, num_classes)
    return alg_obj