# coding: utf-8
import tensorflow as tf
#from theano import tensor as T
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel
from skimage import data, img_as_float
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import structural_similarity
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf 
import math
from IPython.core.debugger import Pdb
pdb = Pdb()
def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
def vifp_mscale(ref, dist):
    sigma_nsq=2
    eps = 1e-10

    num = tf.constant(0,dtype=tf.float32)
    den = tf.constant(0,dtype=tf.float32)
    for scale in range(1, 2):

        N = 2**(4-scale+1) + 1
        sd = N/5.0
        window = _tf_fspecial_gauss(N, sd)
        if (scale >1):
           ref_x = tf.nn.conv2d(ref, window, strides=[1, 1, 1, 1], padding='SAME')
           dist_x = tf.nn.conv2d(dist, window, strides=[1, 1, 1, 1], padding='SAME')
           new_height = tf.round(tf.shape(ref)[1]/scale)
           new_width = tf.round(tf.shape(ref)[2]/scale)
            #print(new_height.eval())
           ref = tf.image.resize_images(ref_x,[new_height,new_width])
            #print(ref2)
            #pdb.set_trace()
           dist = tf.image.resize_images(dist_x,[new_height,new_width])
           # print(ref_x.get_shape())   
        mu1 = tf.nn.conv2d(ref, window, strides=[1, 1, 1, 1], padding='SAME')
        mu2 = tf.nn.conv2d(dist, window, strides=[1, 1, 1, 1], padding='SAME')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(ref*ref, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_sq
        sigma2_sq =  tf.nn.conv2d(dist*dist, window, strides=[1, 1, 1, 1], padding='SAME') - mu2_sq
        sigma12 = tf.nn.conv2d(ref*dist, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_mu2
        sigma1_sq = tf.nn.relu(sigma1_sq)
        sigma2_sq = tf.nn.relu(sigma2_sq)
       
        
        g = tf.reduce_mean(sigma12 / (sigma1_sq + eps))
        sv_sq = sigma2_sq - g * sigma12
       
 
        num=tf.add(num, tf.reduce_sum(tf.log(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq))))
        den=tf.add(den,tf.reduce_sum(tf.log(1 + sigma1_sq / sigma_nsq)))
        ##with tf.Session() as sess:
    		#sess.run(tf.global_variables_initializer())
        #return num
    vifp = (1-tf.divide(num,den))
    vifp2=tf.nn.relu(vifp)
    #if(vifp<0):
    #	vifp=0
    return (vifp2)
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)
def tf_ssim(img1, img2, cs_map=True, mean_metric=False, size=4, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    #print(K.mean(1-value))
    #pdb.set_trace()		
    return value
def tf_ms_ssim(img1, img2, mean_metric=True, level=3):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return tf.reduce_mean(1-value)
class BEGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=64, gamma=0.5):
        self.gamma = gamma
        self.decay_step = 10000
        self.decay_rate = 0.2
        self.beta1 = 0.7
        self.lambd_k = 0.001
        self.nf = 128
        self.lr_lower_bound = 2e-5
        super(BEGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None]+ self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            # Discriminator is not called an energy function in BEGAN. The naming is from EBGAN.
            D_real_energy = self._discriminator(X)
            D_fake_energy = self._discriminator(G, reuse=True)

            k = tf.Variable(0., name='k', trainable=False)
            with tf.variable_scope('D_loss'):
                D_loss = D_real_energy - k * D_fake_energy
            with tf.variable_scope('G_loss'):
                G_loss = D_fake_energy
            with tf.variable_scope('balance'):
                balance = self.gamma*D_real_energy - D_fake_energy
            with tf.variable_scope('M'):
                M = D_real_energy + tf.abs(balance)

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            # The authors suggest decaying learning rate by 0.5 when the convergence mesure stall
            # carpedm20 decays by 0.5 per 100000 steps
            # Heumi decays by 0.95 per 2000 steps (https://github.com/Heumi/BEGAN-tensorflow/)
            D_lr = tf.train.exponential_decay(self.D_lr, global_step, self.decay_step, self.decay_rate, staircase=True)
            D_lr = tf.maximum(D_lr, self.lr_lower_bound)
            G_lr = tf.train.exponential_decay(self.G_lr, global_step, self.decay_step, self.decay_rate, staircase=True)
            G_lr = tf.maximum(G_lr, self.lr_lower_bound)

            with tf.variable_scope('D_train_op'):
                with tf.control_dependencies(D_update_ops):
                    D_train_op = tf.train.AdamOptimizer(learning_rate=D_lr, beta1=self.beta1).\
                        minimize(D_loss, var_list=D_vars)
            with tf.variable_scope('G_train_op'):
                with tf.control_dependencies(G_update_ops):
                    G_train_op = tf.train.AdamOptimizer(learning_rate=G_lr, beta1=self.beta1).\
                        minimize(G_loss, var_list=G_vars, global_step=global_step)

            # It should be ops `define` under control_dependencies
            with tf.control_dependencies([D_train_op]): # should be iterable
                with tf.variable_scope('update_k'):
                    update_k = tf.assign(k, tf.clip_by_value(k + self.lambd_k * balance, 0., 1.)) # define
            D_train_op = update_k # run op

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_energy/real', D_real_energy),
                tf.summary.scalar('D_energy/fake', D_fake_energy),
                tf.summary.scalar('convergence_measure', M),
                tf.summary.scalar('balance', balance),
                tf.summary.scalar('k', k),
                tf.summary.scalar('D_lr', D_lr),
                tf.summary.scalar('G_lr', G_lr)
            ])

            # sparse-step summary
            # Generator of BEGAN does not use tanh activation func.
            # So the generated sample (fake sample) can exceed the image bound [-1, 1].
            fake_sample = tf.clip_by_value(G, -1., 1.)
            tf.summary.image('fake_sample', fake_sample, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('G_hist', G) # for checking out of bound
            # histogram all varibles
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)

            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = fake_sample
            self.global_step = global_step

    def _encoder(self, X, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            nf = self.nf
            nh = self.z_dim

            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(X, nf)

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf*2, stride=2) # 32x32

                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*3, stride=2) # 16x16

                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*4, stride=2) # 8x8

                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)

            net = slim.flatten(net)
            h = slim.fully_connected(net, nh, activation_fn=None)

            return h

    def _decoder(self, h, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            nf = self.nf
            nh = self.z_dim

            h0 = slim.fully_connected(h, 8*8*nf, activation_fn=None) # h0
            net = tf.reshape(h0, [-1, 8, 8, nf])

            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [16, 16]) # upsampling

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [32, 32])

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = tf.image.resize_nearest_neighbor(net, [64, 64])

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)

                net = slim.conv2d(net, 3, activation_fn=None)

            return net

    def _discriminator(self, X, reuse=False):
        learning_phase = K.learning_phase()
        with tf.variable_scope('D', reuse=reuse):
            h = self._encoder(X, reuse=reuse)
            x_recon = self._decoder(h, reuse=reuse)
            hh=tf.placeholder(tf.float32,[None,64,64,3])
            sess = K.get_session()
            #self.__name__ = 'DSSIMObjective'
	    self.kernel_size = 4
	    self.k1 = 0.01
	    self.k2 = 0.03
	    self.max_value = 1.0
	    self.c1 = (self.k1 * self.max_value) ** 2
	    self.c2 = (self.k2 * self.max_value) ** 2
	    self.dim_ordering = K.image_data_format()
	    self.backend = K.backend()
            ######## the proposed loss function is introduced here
            imageA = tf.image.rgb_to_grayscale(X)
            imageB = tf.image.rgb_to_grayscale(x_recon)
	    mssim_index = tf_ms_ssim(imageA, imageB)		
            energy = tf.abs(X-x_recon)
            energy_ssim = tf.reduce_mean(mssim_index)
            energy = tf.reduce_mean(energy)
            return 0.5*energy+0.5*energy_ssim

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            x_fake = self._decoder(z, reuse=reuse)

            return x_fake
