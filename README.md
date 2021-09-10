

[An Orthogonal Classifier for Improving the Adversarial Robustness of Neural Networks]([[2105.09109\] An Orthogonal Classifier for Improving the Adversarial Robustness of Neural Networks (arxiv.org)](https://arxiv.org/abs/2105.09109))



## Usage



```
┌── data # the path of data
│	├── mnist
│	└──cifar10
└── roboc
	├── freeplot # for saving image
	├── infos # for saving trained model
	├── logs # logging the curve of loss, accuracy, robutness
	├── models # Architectures
	├── src
		├── attacks.py # 
		├── base.py # Coach, arranging the training procdure
		├── config.py # You can specify the ROOT as the path of training data.
		├── criteria.py # useful criteria of foolbox
		├── dict2obj.py #
		├── loadopts.py # for loading basic configs
		├── loss_zoo.py # The implementations of loss function ...
		└── utils.py # other usful tools
	├── AT.py # adversarial training, Madry A.
	├── auto_attack.py # Croce F.
	├── requirements.txt # requiresments of packages
	├── STD.py # standing training
	└── white_box_attack.py # the white-box attacks due to foolbox
```



### Training



Use the following codes to adversarially train the model,


    python AT.py resnet32 cifar10 --scale=10 --leverage=0.15
	python AT.py mnist mnist -lp=M --epochs=80 -lr=0.1 -wd=2e-4 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=0.3
	python AT.py mnist fashionmnist -lp=M --epochs=80 -lr=0.1 -wd=2e-4 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=0.3

where leverage is the hyper-parameter alpha in paper.



STD.py is equal to set leverage=1:

```
python STD.py resnet32 cifar10 -lp=STD --epochs=164 -wd=0.0002 --scale=10
python STD.py mnist mnist -lp=M --epochs=80 -lr=0.1 -wd=2e-4 -mom=0.9 -b=128 --scale=20
python STD.py mnist fashionmnist -lp=M --epochs=80 -lr=0.1 -wd=2e-4 -mom=0.9 -b=128 --scale=20
```



### Evaluation


Set the saved path as SP.

    python white_box_attack.py resnet32 cifar10 SP --attack=pgd-linf --epsilon_min=0 --epsilon_max=1 --epsilon_times=20
    python auto_attack.py resnet32 cifar10 SP --norm=Linf --version=standard

