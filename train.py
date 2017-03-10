from __future__ import print_function

import os
import argparse
import math

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np

import chainer
from chainer import cuda, Variable
import chainer.functions as F

from models import Discriminator, GeneratorMNIST, GeneratorCIFAR
from iterators import RandomNoiseIterator, GaussianNoiseGenerator, UniformNoiseGenerator

def get_batch(iter, device_id):
    batch = chainer.dataset.concat_examples(next(iter), device=device_id)
    return Variable(batch)

def update_model(opt, loss):
    opt.target.cleargrads()
    loss.backward()
    opt.update()

def save_ims(filename, ims, dpi=100):

    ims += 1.0
    ims /= 2.0

    if cuda.get_array_module(ims) == cuda.cupy:
        ims = cuda.to_cpu(ims)

    n, c, w, h = ims.shape
    x_plots = math.ceil(math.sqrt(n))
    y_plots = x_plots if n % x_plots == 0 else x_plots - 1
    plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)

    for i, im in enumerate(ims):
        plt.subplot(y_plots, x_plots, i+1)

        if c == 1:
            plt.imshow(im[0], cmap=plt.cm.binary)
        else:
            plt.imshow(im.transpose((1, 2, 0)), interpolation="nearest")

        plt.axis('off')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                            hspace=0)

    plt.savefig(filename, dpi=dpi*2, facecolor='black')
    plt.clf()
    plt.close()

def print_sample(name, noise_samples, opt_generator):
    generated = opt_generator.target(noise_samples)
    save_ims(name, generated.data)
    print("    Saved image to {}".format(name))

def training_step(args, train_iter, test_iter, noise_iter, opt_generator, opt_discriminator):

    noise_samples = get_batch(noise_iter, args.device_id)

    # generate an image
    generated = opt_generator.target(noise_samples)

    # get a batch of the dataset
    train_samples = get_batch(train_iter, args.device_id)

    # update the discriminator
    Dreal = opt_discriminator.target(train_samples)
    Dgen = opt_discriminator.target(generated)

    Dloss = 0.5 * (F.sum((Dreal - 1.0)**2) + F.sum(Dgen**2)) / args.batchsize
    update_model(opt_discriminator, Dloss)

    # update the generator
    noise_samples = get_batch(noise_iter, args.device_id)
    generated = opt_generator.target(noise_samples)
    Gloss = 0.5 * F.sum((opt_discriminator.target(generated) - 1.0)**2) / args.batchsize
    update_model(opt_generator, Gloss)

    if train_iter.is_new_epoch:
        print("[{}] Discriminator loss: {} Generator loss: {}".format(train_iter.epoch, Dloss.data, Gloss.data))
        print_sample(os.path.join(args.output, "epoch_{}.png".format(train_iter.epoch)), noise_samples, opt_generator)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', '-g', type=int, default=-1)
    parser.add_argument('--num_epochs', '-n', type=int, default=100)
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--num_z', '-z', type=int, default=1024)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--output', '-o', type=str, default="output")
    parser.add_argument('--mnist', '-m', action="store_true")
    return parser.parse_args()


def main(args):

    # if we enabled GPU mode, set the GPU to use
    if args.device_id >= 0:
        chainer.cuda.get_device(args.device_id).use()

    # Load dataset (we will only use the training set)
    if args.mnist:
        train, test = chainer.datasets.get_mnist(withlabel=False, scale=2, ndim=3)
        generator = GeneratorMNIST()
    else:
        train, test = chainer.datasets.get_cifar10(withlabel=False, scale=2, ndim=3)
        generator = GeneratorCIFAR()

    # subtracting 1, after scaling to 2 (done above) will make all pixels in the range [-1,1]
    train -= 1.0
    
    num_training_samples = train.shape[0]

    # make data iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # build optimizers and models
    opt_generator = chainer.optimizers.RMSprop(lr=args.learning_rate)
    opt_discriminator = chainer.optimizers.RMSprop(lr=args.learning_rate)

    opt_generator.setup(generator)
    opt_discriminator.setup(Discriminator())

    # make a random noise iterator (uniform noise between -1 and 1)
    noise_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, args.num_z), args.batchsize)

    # send to GPU
    if args.device_id >= 0:
        opt_generator.target.to_gpu()
        opt_discriminator.target.to_gpu()

    # make the output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    print("Starting training loop...")
    
    while train_iter.epoch < args.num_epochs:
        training_step(args, train_iter, test_iter, noise_iter, opt_generator, opt_discriminator)
    
    print("Finished training.")

if __name__=='__main__':
    args = parse_args()
    main(args)