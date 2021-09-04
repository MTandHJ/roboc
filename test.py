


import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("path", type=str)
parser.add_argument("--progress", action="store_false", default=True)
parser.add_argument("--norm", choices=("linf", "l2", "l1"), default=("linf", "l2", "l1"))
opts = parser.parse_args()

fmt = "python white_box_attack.py {model} {dataset} {path} " \
        "--attack={attack} --steps={steps} --stepsize={stepsize} --epsilon_min={epsilon} {progress}"

basic_cfg = dict(
    model=opts.model,
    dataset=opts.dataset,
    path=opts.path,
    progress="--progress" if opts.progress else ""
)

if opts.model == "mnist":
    linf_cfg = (
        dict(attack='pgd-linf', steps=50, stepsize=0.033333, epsilon=0.3),
        dict(attack='pgd-linf', steps=100, stepsize=0.033333, epsilon=0.3),
        dict(attack='deepfool-linf', steps=50, stepsize=0.02, epsilon=0.3),
        dict(attack='fgsm', steps=50, stepsize=0.033333, epsilon=0.3)
    )

    l2_cfg = (
        dict(attack="pgd-l2", steps=100, stepsize=0.05, epsilon=2),
        dict(attack="cw-l2", steps=1000, stepsize=0.01, epsilon=2),
        dict(attack="deepfool-l2", steps=50, stepsize=0.02, epsilon=2),
    )

    l1_cfg = (
        dict(attack="pgd-l1", steps=50, stepsize=0.05, epsilon=10),
        dict(attack="slide", steps=50, stepsize=0.05, epsilon=10),
    )
else:

    linf_cfg = (
        dict(attack="pgd-linf", steps=10, stepsize=0.25, epsilon=8/255),
        dict(attack="pgd-linf", steps=10, stepsize=0.25, epsilon=16/255),
        dict(attack="pgd-linf", steps=20, stepsize=0.1, epsilon=8/255),
        dict(attack="pgd-linf", steps=20, stepsize=0.1, epsilon=16/255),
        dict(attack="pgd-linf", steps=40, stepsize=0.1, epsilon=8/255),
        dict(attack="pgd-linf", steps=40, stepsize=0.1, epsilon=16/255),
        dict(attack="deepfool-linf", steps=50, stepsize=0.02, epsilon=8/255),
        dict(attack="deepfool-linf", steps=50, stepsize=0.02, epsilon=16/255),
        dict(attack="fgsm", steps=50, stepsize=0.02, epsilon=8/255),
        dict(attack="fgsm", steps=50, stepsize=0.02, epsilon=16/255),
    )

    l2_cfg = (
        dict(attack="pgd-l2", steps=50, stepsize=0.1, epsilon=0.5),
        dict(attack="cw-l2", steps=1000, stepsize=0.01, epsilon=0.5),
        dict(attack="deepfool-l2", steps=50, stepsize=0.02, epsilon=0.5),
    )

    l1_cfg = (
        dict(attack="pgd-l1", steps=50, stepsize=0.05, epsilon=12),
        dict(attack="slide", steps=50, stepsize=0.05, epsilon=12),
    )

cfgs = dict(linf=linf_cfg, l2=l2_cfg, l1=l1_cfg)

def main():
    if isinstance(opts.norm, str):
        norms = (opts.norm,)
    else:
        norms = opts.norm
    for norm in norms:
        other_cfg = cfgs[norm]
        for cfg in other_cfg:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print(cfg)
            cfg.update(basic_cfg)
            command = fmt.format(**cfg)
            os.system(command)

if __name__ == "__main__":
    main()