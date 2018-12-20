from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import cv2
import tensorflow as tf
from IPython.core.debugger import Pdb
pdb = Pdb() 
"""
Video Quality Metrics
Copyright (c) 2015 Alex Izvorski <aizvorski@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
NIQE: Natural Image Quality Evaluator

An excellent no-reference image quality metric.  

Code below is roughly similar on Matlab code graciously provided by Anish Mittal, but is written from scratch (since most of the Matlab functions there don't have numpy equivalents).
Training the model is not implemented yet, so we rely on pre-trained model parameters in file modelparameters.mat.

Cite:
Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a completely blind image quality analyzer." Signal Processing Letters, IEEE 20.3 (2013): 209-212.
"""

import math
import numpy
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import scipy.io
import skimage.transform
import tensorflow as tf 
import math
"""
Generalized Gaussian distribution estimation.

Cite: 
Dominguez-Molina, J. Armando, et al. "A practical procedure to estimate the shape parameter in the generalized Gaussian distribution.", 
  available through http://www. cimat. mx/reportes/enlinea/I-01-18_eng. pdf 1 (2001).
"""

"""
Generalized Gaussian ratio function
Cite: Dominguez-Molina 2001, pg 7, eq (8)
"""
def generalized_gaussian_ratio(alpha):
    return (gamma(2.0/alpha)**2) / (gamma(1.0/alpha) * gamma(3.0/alpha))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
"""
Generalized Gaussian ratio function inverse (numerical approximation)
Cite: Dominguez-Molina 2001, pg 13
"""
def generalized_gaussian_ratio_inverse(k):
    a1 = -0.535707356
    a2 = 1.168939911
    a3 = -0.1516189217
    b1 = 0.9694429
    b2 = 0.8727534
    b3 = 0.07350824
    c1 = 0.3655157
    c2 = 0.6723532
    c3 = 0.033834

    if k < 0.131246:
        return 2 * math.log(27.0/16.0) / math.log(3.0/(4*k**2))
    elif k < 0.448994:
        return (1/(2 * a1)) * (-a2 + math.sqrt(a2**2 - 4*a1*a3 + 4*a1*k))
    elif k < 0.671256:
        return (1/(2*b3*k)) * (b1 - b2*k - math.sqrt((b1 - b2*k)**2 - 4*b3*(k**2)))
    elif k < 0.75:
        #print "%f %f %f" % (k, ((3-4*k)/(4*c1)), c2**2 + 4*c3*log((3-4*k)/(4*c1)) )
        return (1/(2*c3)) * (c2 - math.sqrt(c2**2 + 4*c3*math.log((3-4*k)/(4*c1))))
    else:
        print "warning: GGRF inverse of %f is not defined" %(k)
        return numpy.nan

"""
Estimate the parameters of an asymmetric generalized Gaussian distribution
"""
def estimate_aggd_params(x):
    gam=[]
    k=0
    r_gam=[]
    i=0.2
    for kk in range(0,9801):
	gam.append(i)
        r_gam.append((gamma(2.0/i)**2)/(gamma(1/i)*gamma(3/i)))
        k=k+1
        i=i+0.001
    xx=tf.reshape(x,[-1])
    x_
    x_left = tf.cast(xx > 0, xx.dtype)
    print(type(x_left))
    x_right = tf.cast(xx <= 0, xx.dtype)
    print(x_right.get_shape())
    pdb.set_trace()
    stddev_left = math.sqrt((1.0/(x_left.size - 1)) * numpy.sum(x_left ** 2))
    stddev_right = math.sqrt((1.0/(x_right.size - 1)) * numpy.sum(x_right ** 2))
    if stddev_right == 0:
        return 1, 0, 0 # TODO check this
    r_hat = numpy.mean(numpy.abs(x))**2 / numpy.mean(x**2)
    y_hat = stddev_left / stddev_right
    R_hat = r_hat * (y_hat**3 + 1) * (y_hat + 1) / ((y_hat**2 + 1) ** 2)
    ####
    other=(r_gam-R_hat*np.ones(9801))**2
    array_p=np.argmin(other)
    alpha=gam[array_p]
    #alpha = generalized_gaussian_ratio_inverse(R_hat)
    beta_left = stddev_left * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    beta_right = stddev_right * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    return alpha, beta_left, beta_right

def compute_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)

    features.extend([ alpha, (beta_left+beta_right)/2 ])

    for x_shift, y_shift in ((0,1), (1,0), (1,1), (1,-1)):
        img_pair_products  = img_norm * numpy.roll(numpy.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0/alpha) / gamma(1.0/alpha))
        features.extend([ alpha, eta, beta_left, beta_right ])

    return features
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
def normalize_imagetf(img):
    img_norm=tf.image.per_image_standardization(img) 
    size=4
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    mu1 = tf.nn.conv2d(img, window, strides=[size,size,size,size], padding='SAME')
    ########
    mu1_sq = mu1 * mu1
    #sess = tf.InteractiveSession()
    #print(window.eval())
    #pdb.set_trace()
    sigma1_sq = tf.sqrt(tf.abs(tf.nn.conv2d(img*img, window, strides=[1,1,1,1],padding='VALID') - mu1_sq))
    img_norm = (img - mu1) / (sigma1_sq + 1)
    #print(sess.run(img_norm))
    return img_norm
def normalize_image(img, sigma=7/6):
    mu  = gaussian_filter(img, sigma, mode='nearest')
    mu_sq = mu * mu
    print img.shape
    sigma = numpy.sqrt(numpy.abs(gaussian_filter(img * img, sigma, mode='nearest') - mu_sq))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm



def niqe(img):
    model_mat = scipy.io.loadmat('modelparameters.mat')
    model_mu = model_mat['mu_prisparam']
    model_cov = model_mat['cov_prisparam']

    features = None
    img_scaled = img
    for scale in [1,2]:

        #if scale != 1:
            #img_scaled = skimage.transform.rescale(img, 1/scale)
            #img_scaled = scipy.misc.imresize(img_norm, 0.5)
            #pdb.set_trace()
        # print img_scaled
        img_norm = normalize_imagetf(img_scaled)
        print(img_norm)
        #pdb.set_trace()
        scale_features = []
        block_size = 32//scale
        for block_col in range(64//block_size):
            for block_row in range(64//block_size):
                block_values=img_norm[block_col*block_size:(block_col+1)*block_size, block_row*block_size:(block_row+1)*block_size] 
                print(block_values.get_shape())
                print(type(block_values))
                blck=compute_features(block_values)
                #pdb.set_trace()

                block_features = compute_features( img_norm[block_col*block_size:(block_col+1)*block_size, block_row*block_size:(block_row+1)*block_size] )


                scale_features.append(block_features)
        # print "len(scale_features)=%f" %(len(scale_features))
        if (scale==1):
            features = numpy.vstack(scale_features)
            # print features.shape
        else:
            features = numpy.hstack([features, numpy.vstack(scale_features)])
            # print features.shape
        
    features_mu = numpy.mean(features, axis=0)
    features_cov = numpy.cov(features.T)

    pseudoinv_of_avg_cov  = numpy.linalg.pinv((model_cov + features_cov)/2)
    niqe_quality = math.sqrt( (model_mu - features_mu).dot( pseudoinv_of_avg_cov.dot( (model_mu - features_mu).T ) ) )

    return niqe_quality

import sys
img = scipy.misc.imread(sys.argv[1], flatten=True).astype(numpy.float)/255.0
#imageA = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape
#print rows,cols
image1 = tf.placeholder(tf.float32, shape=[rows, cols,1])
#def image_to_4d(image):
   # image = tf.expand_dims(image, 0)
  #  image = tf.expand_dims(image, -1)
#    return image
#image4d_1 = image_to_4d(image1)
#print type(img)
#print type(image4d_1)
niqe2=niqe(image1)
#print niqe2
#pdb.set_trace()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    	tf_niqe_none = sess.run(niqe2,feed_dict={image1: img})
print('tf_niqe', tf_niqe_none)
