import chainer
import chainer.functions as F
import chainer.links as L

class Generator(chainer.Chain):

	def __init__(self):
		super().__init__(
			dc1=L.Deconvolution2D(None, 256, 4, stride=1, pad=0, nobias=True),
			dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, nobias=True),
			dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=2, nobias=True),
			dc4=L.Deconvolution2D(64, 1, 4, stride=2, pad=1, nobias=True),
			bn_dc1=L.BatchNormalization(256),
			bn_dc2=L.BatchNormalization(128),
			bn_dc3=L.BatchNormalization(64)
		)

	def __call__(self, z, test=False):
		h = F.reshape(z, (z.shape[0], -1, 1, 1))
		h = F.relu(self.bn_dc1(self.dc1(h), test=test))
		h = F.relu(self.bn_dc2(self.dc2(h), test=test))
		h = F.relu(self.bn_dc3(self.dc3(h), test=test))
		h = F.tanh(self.dc4(h))
		return h

class Discriminator(chainer.Chain):

	def __init__(self):
		super(Discriminator, self).__init__(
		c0 = L.Convolution2D(None, 64, 4, stride=2, pad=1, nobias=True),
		c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, nobias=True),
		c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, nobias=True),
		c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, nobias=True),
		l4l = L.Linear(None, 1),
		bn0 = L.BatchNormalization(64),
		bn1 = L.BatchNormalization(128),
		bn2 = L.BatchNormalization(256),
		bn3 = L.BatchNormalization(512),
		)
		
	def __call__(self, x, test=False):
		h = F.leaky_relu(self.c0(x))   
		h = F.leaky_relu(self.bn1(self.c1(h), test=test))
		h = F.leaky_relu(self.bn2(self.c2(h), test=test))
		h = F.leaky_relu(self.bn3(self.c3(h), test=test))
		l = self.l4l(h)
		return l
