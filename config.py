from models import *
import tensorflow as tf


model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'EBGAN', 'BEGAN', 'DRAGAN', 'CoulombGAN']

def get_model(mtype, name, training):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'DRAGAN':
        model = dragan.DRAGAN
    elif mtype == 'COULOMBGAN':
        model = coulombgan.CoulombGAN
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)

#### If there are multiple datasets there we can keep multiple folders and paths
def get_dataset(dataset_name):
    celebA_128 = './data/celebA_128_tfrecords/*.tfrecord'
    car_128='./data/car/car_128_tfrecords/*.tfrecord'
    if dataset_name == 'celeba':
        path = celebA_128
        n_examples = 202599
    elif dataset_name == 'car':
        path = car_128
        n_examples = 16185
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))

    return path, n_examples


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

