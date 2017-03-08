from __future__ import print_function

from matplotlib import pyplot as plt

import argparse

import chainer
from chainer import cuda, Variable
import chainer.functions as F

import numpy as np

from models import Discriminator, Generator
from iterators import RandomNoiseIterator, GaussianNoiseGenerator

def get_batch(iter, device_id):
	#chainer.dataset.concat_examples
		
	#batch = Variable(iter.next(), device=device_id) 
	batch = next(iter)
	batch = chainer.dataset.concat_examples(batch, device=device_id)
	#print(batch.shape)
	return Variable(batch)

def update_model(opt, loss):
	opt.target.cleargrads()
	loss.backward()
	opt.update()

def save_ims(filename, ims, dpi=100):

    n, c, w, h = ims.shape
    x_plots = math.ceil(math.sqrt(n))
    y_plots = x_plots if n % x_plots == 0 else x_plots - 1
    plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)

    for i, im in enumerate(ims):
        plt.subplot(y_plots, x_plots, i+1)

        if c == 1:
            plt.imshow(im[0])
        else:
            plt.imshow(im.transpose((1, 2, 0)), interpolation="nearest")

        plt.axis('off')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        #plt.gray()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                            hspace=0)

    plt.savefig(filename, dpi=dpi*2, facecolor='black')
    plt.clf()
    plt.close()

def get_sample(name, noise_iter, opt_generator, device_id):

	noise_samples = get_batch(noise_iter, args.device_id)
	generated = opt_generator.target(noise_samples)

	save_ims(name, generated)

def training_step(args, train_iter, test_iter, noise_iter, opt_generator, opt_discriminator):
	
	# generate some noise  
	#noise_samples = Variable(cuda.cupy.random.uniform(-1, 1, (args.batchsize, args.num_z), dtype=np.float32))

	noise_samples = get_batch(noise_iter, args.device_id)
	#print(noise_samples.shape)

	# generate an image
	generated = opt_generator.target(noise_samples)
	#print("generated image shape {}".format(generated.shape))

	# get a batch of the dataset
	train_samples = get_batch(train_iter, args.device_id)

	# update the discriminator
	Dreal = opt_discriminator.target(train_samples)
	Dgen = opt_discriminator.target(generated)

	Dloss = 0.5 * (F.sum((Dreal - 1)**2) + F.sum(Dgen**2)) / args.batchsize
	update_model(opt_discriminator, Dloss)

	# update the generator
	noise_samples = get_batch(noise_iter, args.device_id)
	generated = opt_generator.target(noise_samples)
	Gloss = 0.5 * F.sum((opt_discriminator.target(generated) - 1)**2) / args.batchsize
	update_model(opt_generator, Gloss)

	print("{} {}".format(Dloss.data, Gloss.data))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device_id', '-g', type=int, default=-1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batchsize', type=int, default=64)
	parser.add_argument('--num_z', '-z', type=int, default=256)
	"""parser.add_argument('--resume', '-r', type=str, default=None)
	parser.add_argument('--output', '-o', type=str, default='result')
	parser.add_argument('--procs', '-p', type=int, default=12)
	parser.add_argument('--sample_img', '-s', type=bool, default=True)
	parser.add_argument('--regularization_strength', '-rs', type=float, default=0.1)
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
	parser.add_argument('--clipping_parameter', '-c', type=float, default=0.01)
	parser.add_argument("--history_buffer", "-hb", action="store_true")"""
	return parser.parse_args()


def main(args):
	"""device_id = 0
	num_epochs = 100
	batchsize = 64
	num_z = 256"""
	#print(args.batchsize)
	# Load the MNIST dataset
	train, test = chainer.datasets.get_mnist(withlabel=False, ndim=3)
	num_training_samples = train.shape[0]

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

	# build model and optimizers
	opt_generator = chainer.optimizers.RMSprop(lr=0.00005)
	opt_discriminator = chainer.optimizers.RMSprop(lr=0.00005)

	opt_generator.setup(Generator())
	opt_discriminator.setup(Discriminator())

	# make a random noise iterator
	noise_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, args.num_z), args.batchsize)

	# send to GPU
	if args.device_id >= 0:
		opt_generator.target.to_gpu()
		opt_discriminator.target.to_gpu()

	# start training loop
	for epoch in range(args.num_epochs):
		for batch in range(num_training_samples // args.batchsize):
			training_step(args, train_iter, test_iter, noise_iter, opt_generator, opt_discriminator)
		#get_sample("epoch_{}.png".format(epoch), noise_iter, opt_generator, device_id)

if __name__=='__main__':
	args = parse_args()
	main(args)