


from typing import Callable, TypeVar, Any, Union, Optional, List, Tuple, Dict, Iterable, cast
import torch
import torch.nn as nn
import foolbox as fb
import eagerpy as ep
import os

from models.base import AdversarialDefensiveModel
from .criteria import LogitsAllFalse
from .utils import AverageMeter, ProgressMeter
from .loss_zoo import cross_entropy, kl_divergence
from .config import SAVED_FILENAME


def enter_attack_exit(func) -> Callable:
    def wrapper(attacker: "Adversary", *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Coach:
    
    def __init__(
        self, model: AdversarialDefensiveModel,
        device: torch.device,
        loss_func: Callable, 
        normalizer: Callable[[torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy"
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc.")
        self.progress = ProgressMeter(self.loss, self.acc)
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    def train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        *, epoch: int = 8888
    ) -> float:

        self.progress.step() # reset the meter
        self.model.train()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.train() # make sure in training mode
            features, logits = self.model(self.normalizer(inputs))
            loss = self.loss_func(features, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (logits.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0))
            self.acc.update(accuracy_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.loss.avg

    def _get_targets(self, labels):
        weights = self.model.fc.weight.data.clone()
        targets = weights[labels]
        return targets

    def adv_train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        attacker: "Adversary", 
        *, leverage: float = 0.15, epoch: int = 8888
    ) -> float:
    
        assert isinstance(attacker, Adversary)
        self.progress.step() # reset the meter
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            criterion = LogitsAllFalse(self._get_targets(labels)) # perturbed by squared loss
            _, clipped, _ = attacker(inputs, criterion)
            
            self.model.train()
            features_nat, logits_nat = self.model(self.normalizer(inputs))
            features_adv, logits_adv = self.model(self.normalizer(clipped))
            loss_nat = self.loss_func(features_nat, labels)
            loss_adv = self.loss_func(features_adv, labels)
            loss = leverage * loss_nat + (1 - leverage) * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0))
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step() # update the learning rate
        return self.loss.avg



class FBDefense:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        bounds: Tuple[float, float], 
        preprocessing: Optional[Dict]
    ) -> None:
        self.rmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device            
        )

        self.model = model
    
    def train(self, mode: bool = True) -> None:
        self.model.train(mode=mode)

    def eval(self) -> None:
        self.train(mode=False)

    def query(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rmodel(inputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.query(inputs)


class Adversary:
    """
    Adversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model: AdversarialDefensiveModel, 
        attacker: Callable, device: torch.device,
        bounds: Tuple[float, float], 
        preprocessing: Optional[Dict], 
        epsilon: Union[None, float, List[float]]
    ) -> None:

        model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    def attack(
        self, 
        inputs: torch.Tensor, 
        criterion: Any, 
        epsilon: Union[None, float, List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluating mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    def __call__(
        self, 
        inputs: torch.Tensor, criterion: Any,
        epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.attack(inputs, criterion, epsilon)


class AdversaryForTrain(Adversary):

    @enter_attack_exit
    def attack(
        self, inputs: torch.Tensor, 
        criterion: Any, 
        epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return super(AdversaryForTrain, self).attack(inputs, criterion, epsilon)


class AdversaryForValid(Adversary): 

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor) -> int:
        inputs_, labels_ = ep.astensors(inputs, labels)
        del inputs, labels

        self.model.eval() # make sure in evaluating mode ...
        predictions = self.fmodel(inputs_).argmax(axis=-1)
        accuracy = (predictions == labels_)
        return cast(int, accuracy.sum().item())

    def success(
        self, 
        inputs: torch.Tensor, criterion: Any, 
        epsilon: Union[None, float, List[float]] = None
    ) -> int:

        _, _, is_adv = self.attack(inputs, criterion, epsilon)
        return cast(int, is_adv.sum().item())

    def evaluate(
        self, 
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        epsilon: Union[None, float, List[float]] = None
    ) -> Tuple[float, float]:

        datasize = len(dataloader.dataset) # type: ignore
        running_accuracy = 0
        running_success = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            running_accuracy += self.accuracy(inputs, labels)
            running_success += self.success(inputs, labels, epsilon)
        return running_accuracy / datasize, running_success / datasize



