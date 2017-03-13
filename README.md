# chainer-LSGAN
An implementation of [_Mao et al., "Least Squares Generative Adversarial Networks" 2017_](https://arxiv.org/abs/1611.04076) using the [Chainer framework](http://chainer.org/). 

CIFAR10 & MNIST for 100 epochs
-------
<p align="center">
  <img src="images/CIFAR10_epoch100.png" height="480" width="480" alt="CIFAR10"/> <img src="images/MNIST_epoch100.png" height="480" width="480" alt="MNIST"/>
</p>

Usage
-------
Trains on the CIFAR10 dataset by default, and will generate an image of a sample batch from the network after each epoch. Run the following:
```
python train.py --device_id 0
```
to train. By default, an output folder will be created in your current working directory. Setting `--device_id` to -1 will run in CPU mode, whereas 0 will run on GPU number 0 etc. To train on MNIST, use the flag `--mnist`.

License
-------
MIT License. Please see the LICENSE file for details.
