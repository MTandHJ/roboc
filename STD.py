#!/usr/bin/env python


from typing import Tuple
import argparse
from src.loadopts import *



METHOD = "RobOC-STD"
SAVE_FREQ = 5
PRINT_FREQ = 20
FMT = "{description}={scale}" \
        "{learning_policy}-{optimizer}-{lr}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)

# for orthogonal classifier
parser.add_argument("--scale", type=float, default=10.,
                help="the length of weights")

# basic settings
parser.add_argument("--loss", type=str, default="square")
parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=2e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
parser.add_argument("-lp", "--learning_policy", type=str, default="STD", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=164)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied during training.")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("-m", "--description", type=str, default="RobOC-STD")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)




def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import Coach
    from src.utils import gpu, set_seed, load_checkpoint

    cfg = Config()
    set_seed(opts.seed)

    # the model and other settings for training
    model = load_model(opts.model)(
        num_classes=get_num_classes(opts.dataset),
        scale=opts.scale
    )
    device = gpu(model)

    # load the dataset
    trainset = load_dataset(
        dataset_type=opts.dataset, 
        transform=opts.transform, 
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset, 
        batch_size=opts.batch_size, 
        train=True,
        show_progress=opts.progress
    )
    testset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=False
    )
    cfg['testloader'] = load_dataloader(
        dataset=testset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )
    normalizer = load_normalizer(dataset_type=opts.dataset)

    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optimizer, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, 
        learning_policy_type=opts.learning_policy,
        T_max=opts.epochs
    )

    # generate the path for logging information and saving parameters
    cfg['info_path'], cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        model=opts.model, description=opts.description
    )
    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    cfg['coach'] = Coach(
        model=model, device=device, 
        loss_func=load_loss_func(opts.loss)(model=model), 
        normalizer=normalizer, optimizer=optimizer,
        learning_policy=learning_policy
    )

    # for validation
    cfg['valider'] = load_valider(
        model=model, device=device, 
        dataset_type=opts.dataset
    )
    return cfg


def evaluate(
    valider, trainloader, testloader,
    acc_logger, rob_logger, writter,
    epoch = 8888
):
    train_accuracy, train_success = valider.evaluate(trainloader)
    valid_accuracy, valid_success = valider.evaluate(testloader)
    print(f"Train >>> [TA: {train_accuracy:.5f}]    [RA: {1 - train_success:.5f}]")
    print(f"Test. >>> [TA: {valid_accuracy:.5f}]    [RA: {1 - valid_success:.5f}]")
    writter.add_scalars("Accuracy", {"train":train_accuracy, "valid":valid_accuracy}, epoch)
    writter.add_scalars("Success", {"train":train_success, "valid":valid_success}, epoch)

    acc_logger.train(data=train_accuracy, T=epoch)
    acc_logger.valid(data=valid_accuracy, T=epoch)
    rob_logger.train(data=1 - train_success, T=epoch)
    rob_logger.valid(data=1 - valid_success, T=epoch)


def main(
    coach, valider, 
    trainloader, testloader, start_epoch, 
    info_path, log_path
):  
    from src.utils import save_checkpoint, TrackMeter, ImageMeter
    from src.dict2obj import Config
    acc_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )
    acc_logger.plotter = ImageMeter(*acc_logger.values(), title="Accuracy")

    rob_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )
    rob_logger.plotter = ImageMeter(*rob_logger.values(), title="Robustness")


    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        if epoch % PRINT_FREQ == 0:
            evaluate(
                valider=valider, trainloader=trainloader, testloader=testloader,
                acc_logger=acc_logger, rob_logger=rob_logger, writter=writter,
                epoch=epoch
            )
            

        running_loss = coach.train(trainloader, epoch=epoch)
        writter.add_scalar("Loss", running_loss, epoch)


    evaluate(
        valider=valider, trainloader=trainloader, testloader=testloader,
        acc_logger=acc_logger, rob_logger=rob_logger, writter=writter,
        epoch=opts.epochs
    )

    acc_logger.plotter.plot()
    rob_logger.plotter.plot()
    acc_logger.plotter.save(writter)
    rob_logger.plotter.save(writter)


if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg = load_cfg()
    mkdirs(cfg.info_path, cfg.log_path)
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=cfg.log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()










