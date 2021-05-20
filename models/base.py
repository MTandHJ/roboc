

import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

class ADType(abc.ABC): ...

class AdversarialDefensiveModel(ADType, nn.Module):
    """
    Define some basic properties.
    """
    def __init__(self) -> None:
        super(AdversarialDefensiveModel, self).__init__()
        # Some model's outputs for training(evaluating) 
        # and attacking are different.
        self.attacking: bool = False

        
    def attack(self, mode: bool = True) -> None:
        # enter attacking mode
        self.attacking = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.attack(mode)


def generate_weights(length):
    w = torch.tensor([[1.]])
    while True:
        assert length is 1 or length % 2 is 0, \
            "dim_feature should be the form of 2^N."
        length = length // 2
        if length is 0:
            break
        w1 = torch.cat((w, w), dim=1)
        w2 = torch.cat((-w, w), dim=1)
        w = torch.cat((w1, w2), dim=0)
    return F.normalize(w, dim=1)


if __name__ == "__main__":
    
    model = AdversarialDefensiveModel()
    model.child1 = AdversarialDefensiveModel()
    model.child2 = AdversarialDefensiveModel()

    print(model.attack)
    model.attack()
    for m in model.children():
        print(m.attacking)

    model.defense()
    for m in model.children():
        print(m.attacking)

