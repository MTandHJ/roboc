#!/usr/bin/env python

from typing import Tuple
import torch
import argparse
from src.loadopts import *
from models.base import AdversarialDefensiveModel
from autoattack import AutoAttack



METHOD = "AutoAttack"
FMT = "{description}={norm}-{version}-{epsilon}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)

# for AA
parser.add_argument("--norm", choices=("Linf", "L2"), default="Linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--version", choices=("standard", "plus"), default="standard")
parser.add_argument("-b", "--batch_size", type=int, default=128)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)


class Defense(AdversarialDefensiveModel):
    """
    The inputs should be normalized 
    before fed into the model.
    """
    def __init__(
        self, model: AdversarialDefensiveModel, normalizer: "_Normalize"
    ):
        super(Defense, self).__init__()

        self.model = model
        self.normalizer = normalizer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_ = self.normalizer(inputs)
        return self.model(inputs_)


def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.utils import gpu, load, set_seed

    cfg = Config()
    set_seed(opts.seed)

    # load the model
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    device = gpu(model)
    load(
        model=model, 
        path=opts.info_path,
        device=device
    )
    model.eval()

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset, 
        transform='None',
        train=False
    )
    data = []
    targets = []
    for i in range(len(testset)):
        img, label = testset[i]
        data.append(img)
        targets.append(label)
    
    cfg['data'] = torch.stack(data)
    cfg['targets'] = torch.tensor(targets, dtype=torch.long)
    normalizer = load_normalizer(opts.dataset)

    # generate the log path
    _, log_path = generate_path(METHOD, opts.dataset, 
                        opts.model, opts.description)

    cfg['attacker'] = AutoAttack(
        Defense(model, normalizer),
        norm=opts.norm,
        eps=opts.epsilon,
        version=opts.version,
        device=device
    )

    return cfg, log_path


def main(attacker, data, targets):
    attacker.run_standard_evaluation(data, targets, bs=opts.batch_size)



if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()


