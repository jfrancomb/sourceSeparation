from numpy.lib.utils import source
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import callbacks
import tensorflow_addons as tfa
from tensorflow.python.keras.backend import conv1d, sigmoid
from tensorflow.keras.losses import MSE
import tfdata_io
import numpy as np
import os
import datetime
from scipy.io.wavfile import write
import models


# training parameters 
params = {
    'sr'                    : 16000,
    'mixture_seconds'       : 10,
    'nsamples'              : 16000 * 10,
    'n_sources'             : 4,
    'batch_size'            : 7,
    'prefetch_buffer_size'  : 512
}
encoder_params = {
    'n_filters'         : 512,
    'window_size'       : 32,
    'window_hop'        : 16,
}
separator_params = {
    'n_hidden_filters'  : 512,
    'BN_output_dim'     : 128,
    'n_layers'          : 8,
    'n_stacks'          : 3,
    'kernel_size'       : 3,
    'output_dim'        : 512 * params['n_sources']
}
decoder_params = {
    'n_filters'         : 1,
    'kernel_size'       : 32,
    'strides'           : 16
}

data_dir = '/home/jim/projects/sound-separation/datasets/fuss/data/fuss_dev/ssdata_reverb/'
train_list = os.path.join(data_dir, 'train_example_list.txt')
dataset_train = tfdata_io.wavs_to_dataset(file_list = train_list,
                                             nsamples = params['nsamples'],
                                             batch_size = params['batch_size'],
                                             prefetch_buffer_size = params['prefetch_buffer_size'],
                                             shuffle_data = False,
                                             num_examples = -1,
                                             shuffle_buffer_size = 50,
                                             repeat = False)

wav = dataset_train.take(1)
example = iter(dataset_train).next()
mix = example[0][0,:,:].numpy()
sources = example[1][0,:,:].numpy()

write("wavs/mixture.wav", 16000, mix)
write("wavs/source0.wav", 16000, sources[0,:])
write("wavs/source1.wav", 16000, sources[1,:])
write("wavs/source2.wav", 16000, sources[2,:])
write("wavs/source3.wav", 16000, sources[3,:])

mixture = next(iter(dataset_train))[0]
sources = next(iter(dataset_train))[1]


enc = models.ConvEncoder(params=encoder_params, input_signal_shape=mixture.shape)
enc_output = enc(mixture)

sep = models.TemporalConvNet(separator_params, input_signal_shape=enc_output.shape)
sep_output = sep(enc_output)

masks = tf.keras.activations.sigmoid(sep_output)

print('bye')
