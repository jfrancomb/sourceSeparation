from numpy.lib.utils import source
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import callbacks
from tensorflow.python.keras.layers.convolutional import Conv1D
import tensorflow_addons as tfa
from tensorflow.python.keras.backend import conv1d, sigmoid
import tfdata_io
import os
from tensorflow.keras.losses import MSE
import datetime

def model_fn(params, encoder_params, separator_params, decoder_params):

    data_dir = '/home/jim/projects/sound-separation/datasets/fuss/data/fuss_dev/ssdata_reverb/'
    train_list = os.path.join(data_dir, 'train_example_list.txt')

    print('reading dataset from files')
    dataset_train = tfdata_io.wavs_to_dataset(file_list = train_list,
                                             nsamples = params['nsamples'],
                                             batch_size = params['batch_size'],
                                             prefetch_buffer_size = params['prefetch_buffer_size'],
                                             shuffle_data = False,
                                             num_examples = -1,
                                             shuffle_buffer_size = 50,
                                             repeat = False)

    data_dir = '/home/jim/projects/sound-separation/datasets/fuss/data/fuss_dev/ssdata_reverb/'
    test_list = os.path.join(data_dir, 'eval_example_list.txt')
    dataset_test = tfdata_io.wavs_to_dataset(file_list = test_list,
                                             nsamples = params['nsamples'],
                                             batch_size = params['batch_size'],
                                             prefetch_buffer_size = params['prefetch_buffer_size'],
                                             shuffle_data = False,
                                             num_examples = -1,
                                             shuffle_buffer_size = 50,
                                             repeat = False)

    #callbacks for training checkpoints and tensorboard
    tb_path = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1)
    
    checkpoint_filepath = 'checkpoints/checkpoint'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)

    print('running model')
    tasNet = TasNet(params, encoder_params, separator_params, decoder_params)
    tasNet.compile(optimizer = 'adam', loss = "mean_squared_error")
    mixture = next(iter(dataset_train))[0]
    out = tasNet(mixture)

    tasNet.fit(x = dataset_train.take(1), validation_data=dataset_test.take(5),
               callbacks = [checkpoint_callback, tb_callback], epochs = 1)

    tf.keras.utils.plot_model(
        tasNet,
        to_file="zmodel.png",
        expand_nested=True
    )

    print('bye')


class TasNet(tf.keras.Model):
    def __init__(self, params, encoder_params, separator_params, decoder_params):
        # need to initialize itself. not a real super, since there's no pre-defined TasNet
        # this gives you keras.Model attributes and methods, I think
        super(TasNet, self).__init__()
        self.params = params
        self.encoder_params = encoder_params
        self.separator_params = separator_params
        self.decoder_params = decoder_params
        self.mixture_shape = (None, params['nsamples'], 1)
        # Need to add to the input signal length to account for padding
        pad_size = self.get_pad_size(input_signal_shape = self.mixture_shape)
        self.mixture_shape = (None, int(self.mixture_shape[1] + pad_size), self.mixture_shape[2])
        # Convolutional encoder params
        self.encoder = ConvEncoder(encoder_params, input_signal_shape = self.mixture_shape)
        self.encoder.build(input_shape = self.mixture_shape)
        enc_output_shape = self.encoder.get_output_shape()
        # TemporalConvNet params
        self.separator = TemporalConvNet(separator_params, input_signal_shape=enc_output_shape)
        self.separator.build(input_shape = enc_output_shape)
        sep_output_shape = self.separator.get_output_shape()
        # Mask shape needs to be adjusted since the TCN returns all sources appended to each other 
        # This separates the sources dimension so it conforms with the encoder outputs dimensions
        mask_shape=[sep_output_shape[0], self.params['n_sources'], sep_output_shape[1], 
                    int(sep_output_shape[2] / self.params['n_sources'])]
        # The transposed convolution expects a 3 dimensional input, 
        # so the batch and source dimensions are combined for the transposed convolution.
        decoder_input_shape = (None , mask_shape[2], mask_shape[3])
        self.decoder = TransposedConvDecoder(decoder_params, input_signal_shape=decoder_input_shape)
        self.decoder.build(input_shape = decoder_input_shape)
    
    def call(self, input):
        input, rest = self.pad_signal(input)
        # encode signal 
        enc_output = self.encoder(input)
        # separate output 
        sep_output = self.separator(enc_output)
        # calculate masks 
        masks = tf.keras.activations.sigmoid(sep_output)
        masks_reshaped = tf.reshape(masks, shape=[sep_output.shape[0], sep_output.shape[1], self.params['n_sources'], 
                                    int(sep_output.shape[2] / self.params['n_sources'])])
        masks_reshaped = tf.transpose(masks_reshaped, perm=[0,2,1,3])
        ### validate reshaping. 
        # I am leaving this here because I think it is illustrative of the transformation
        # print(masks[0,0:5,0:5])
        # print(masks_reshaped[0,0,0:5,0:5])
        # print(masks[0,0:5, 512:517])
        # print(masks_reshaped[0,1,0:5,0:5])
        #### end validation
        enc_output_reshaped = tf.reshape(enc_output, [enc_output.shape[0], 1, enc_output.shape[1], enc_output.shape[2]])
        masked_output = enc_output_reshaped * masks_reshaped
        # decode masked signal
        # the transposed convolution layer is expecting a 3 dimensional input instead of 4 like the masked output
        # The batch dimension and source dimension are combined
        decoder_input = tf.reshape(masked_output, (masked_output.shape[0] * masked_output.shape[1], 
                                                   masked_output.shape[2], masked_output.shape[3]))

        ### Validate reshaping 
        # print(masked_output.shape)
        # print(decoder_input.shape)
        # print(masked_output[0,0, 0:5,0:5])
        # print(decoder_input[0, 0:5,0:5])
        # print(masked_output[1, 0, 0:5, 0:5])
        # print(decoder_input[4, 0:5, 0:5])
        # print(masked_output[0, 2, 0:5,0:5])
        # print(decoder_input[2, 0:5, 0:5])
        ### end validation

        # Decode
        output = self.decoder.forward(decoder_input)
        # un-zero-pad
        output = output[:,self.encoder_params['window_hop']:-(rest+self.encoder_params['window_hop']), : ]
        output_reshaped = tf.reshape(output, (self.params['batch_size'], self.params['n_sources'], -1, 1))
        output_reshaped = tf.squeeze(output_reshaped, 3)
        ### validating reshaping
        # print(output.shape)
        # print(output_reshaped.shape)
        # print(output[0, 0:5, 0])
        # print(output_reshaped[0,0, 0:5])
        # print(output[4, 0:5, 0])
        # print(output_reshaped[1, 0, 0:5])
        # print(output[3, 0:5, 0])
        # print(output_reshaped[0, 3, 0:5])
        ### validating reshaping
        return output_reshaped
    
    def train_step(self, data):
        x = data[0]
        y = data[1]
        with tf.GradientTape() as tape:
            y_pred = self(x)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            print(self.losses)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            print('at a loss')
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def pad_signal(self, input):
        batch_size = input.shape[0]
        nsample = input.shape[1]
        stride = self.encoder_params['window_hop']
        window = self.encoder_params['window_size']
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            # zero pad pad the end of the signal 
            pad = tf.zeros((batch_size, rest, 1))
            input = tf.concat([input,pad],1)
        # zero pad both the beginning and end of the signal by stride/hop length
        pad_aux = tf.zeros((batch_size, stride, 1))
        input = tf.concat([pad_aux, input, pad_aux], 1)
        return input, rest

    def get_pad_size(self, input_signal_shape):
        nsample = input_signal_shape[1]
        window = self.encoder_params['window_size']
        stride = self.encoder_params['window_hop']
        rest = window - (stride + nsample % window) % window
        pad_size = 0
        if rest > 0:
            pad_size +=  rest
        pad_size += stride * 2
        return pad_size

# input encoder 
class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self, params, input_signal_shape, name=None):
        super(ConvEncoder, self).__init__()
        print('Creating new Convolutional Encoder')
        super().__init__(name=name)
        self.input_signal_shape = input_signal_shape
        self.params = params
        n_filters = params['n_filters']
        window_size = params['window_size']
        window_hop = params['window_hop']
        self.conv_encoder = tf.keras.layers.Conv1D(
            filters = n_filters, kernel_size = window_size,
            strides = window_hop, input_shape = input_signal_shape)

    def get_output_shape(self):
        output_chan = self.params['n_filters']
        output_len = (self.input_signal_shape[1] / self.params['window_hop']) - 1
        return (None, int(output_len), output_chan) 

    def call(self, input):
        encoded_signal = self.conv_encoder(input)
        return encoded_signal

# output decoder
class TransposedConvDecoder(tf.keras.layers.Layer):
    def __init__(self, params, input_signal_shape, name=None):
        super(TransposedConvDecoder, self).__init__()
        self.input_signal_shape = input_signal_shape
        self.params = params
        n_filters = params['n_filters']
        kernel_size = params['kernel_size']
        strides = params['strides']
        #self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        self.tconv_layer = tf.keras.layers.Conv1DTranspose(
            filters = n_filters, kernel_size = kernel_size, use_bias=False,
            strides = strides, padding = 'valid', input_shape = input_signal_shape)

    def forward(self, input):
        decoded_signal = self.tconv_layer(input)
        return decoded_signal

class TemporalConvNet(tf.keras.layers.Layer):
    def __init__(self, params, input_signal_shape, name = None):
        super(TemporalConvNet, self).__init__()
        self.input_signal_shape = input_signal_shape
        self.params = params
        #Norm Params
        BN_output_dim=params['BN_output_dim']#128,
        #DCN params
        n_layers=params['n_layers'] #8, 
        n_stacks = params['n_stacks'] #=3, 
        n_hidden_filters = params['n_hidden_filters'] #=512, 
        kernel_size = params['kernel_size'] #=3,
        # output params
        output_dim = params['output_dim'] #128,
        self.LN = tfa.layers.GroupNormalization(groups=1, epsilon=1e-8)
        # if causal use the cLN... my tf implementation needs work.. don't use causal for now
        #self.LN = cLN(input_dim, eps=1e-8)
        self.BN = tf.keras.layers.Conv1D(filters = BN_output_dim, kernel_size=1, strides = 1,
                                         input_shape = (None, input_signal_shape[1], input_signal_shape[2]))
        # TCN for feature extraction
        self.receptive_field = 0
        #self.TCN = nn.ModuleList([])
        self.TCN_stacks = []
        for s in range(n_stacks):
            for i in range(n_layers):
                self.TCN_stacks.append(
                    DepthwiseSeparableConv(
                        n_hidden_filters = n_hidden_filters, 
                        kernel_size = kernel_size, 
                        dilation = 2**i, 
                        pad_type = 'causal',
                        input_channels = BN_output_dim,
                        input_len = input_signal_shape[1]))
                if i == 0 and s == 0:
                    self.receptive_field += kernel_size
                else:
                    self.receptive_field += (kernel_size - 1) * 2**i
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))   
        # output layers
        self.output_prelu = tf.keras.layers.PReLU()
        self.output_conv = tf.keras.layers.Conv1D(filters=output_dim, kernel_size = 1, strides = 1)

    def get_output_shape(self):
        out_len = self.input_signal_shape[1]
        out_chan = self.params['output_dim']
        return (None, out_len, out_chan)

    def call(self, input):        
        # normalization
        ln_output = self.LN(input)
        bn_output = self.BN(ln_output)
        output = bn_output
        skip_connection = 0.
        for i in range(len(self.TCN_stacks)):
            dsc = self.TCN_stacks[i]
            residual, skip = dsc(output)
            output = output + residual
            skip_connection = skip_connection + skip
        tcn_pre_output = self.output_prelu(skip_connection)
        tcn_output = self.output_conv(tcn_pre_output)
        return tcn_output

class DepthwiseSeparableConv(tf.keras.layers.Layer):
    def __init__(self, n_hidden_filters = 512, kernel_size = 3, 
                dilation = 1, pad_type = 'causal',
                input_channels = 128, input_len = 9999):
        super(DepthwiseSeparableConv, self).__init__()
        # conv layer 1
        # conv layer 1 is just a regular conv1d layer with stride=1
        self.conv_hidden =  tf.keras.layers.Conv1D(filters = n_hidden_filters, kernel_size=1, 
                                                    strides = 1, #dilation_rate=dilation,
                                                    input_shape = (None, input_len , input_channels))
        self.conv_prelu = tf.keras.layers.PReLU()
        self.conv_norm = tfa.layers.GroupNormalization(groups=1, epsilon=1e-8)
        # conv layer 2
        # The kernel_size and dilation are only applied in this layer. 
        self.dconv_hidden = tf.keras.layers.Conv1D(filters = n_hidden_filters, kernel_size=kernel_size, 
                                                    strides = 1, dilation_rate = dilation, padding='causal',
                                                    input_shape = (None, input_len, n_hidden_filters))
        self.dconv_prelu = tf.keras.layers.PReLU()
        self.dconv_norm = tfa.layers.GroupNormalization(groups=1, epsilon=1e-8)
        # outputs
        self.res_out_layer = tf.keras.layers.Conv1D(filters = input_channels, kernel_size=1, strides=1,
                                                    input_shape = (None, input_len , n_hidden_filters))               
        self.skip_out_layer = tf.keras.layers.Conv1D(filters = input_channels, kernel_size=1, strides=1,
                                                     input_shape = (None, input_len , n_hidden_filters))

    def call(self, dsc_input):
        # layer 1
        hidden_out_1 = self.conv_hidden(dsc_input)
        hidden_act_1 = self.conv_prelu(hidden_out_1)
        conv_norm_1 = self.conv_norm(hidden_act_1)
        # layer 2
        hidden_out_2 = self.dconv_hidden(conv_norm_1)
        hidden_act_2 = self.conv_prelu(hidden_out_2)
        conv_norm_2 = self.conv_norm(hidden_act_2)
        #outputs
        res_out = self.res_out_layer(conv_norm_2)
        skip_out = self.skip_out_layer(conv_norm_2)
        return res_out, skip_out




# Cumulative Layer Normalization
# class cLN(tf.keras.layers.Layer):
#     def __init__(self, dimension, eps = 1e-8, trainable=True):
#         super(cLN, self).__init__()
#         self.eps = eps
#         #if trainable:
#         #        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
#         #        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
#         #else:
#         #        self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
#         #        self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)
#         self.gain = tf.Variable(tf.ones(1, dimension,1), trainable = trainable)
#         self.bias = tf.Variable(tf.zeros(1, dimension, 1), trainable = trainable)

#     def forward(self, input):
#         # input size: (Batch, Freq, Time)
#         # cumulative mean for each time step

#         batch_size = input.size(0)
#         channel = input.size(1)
#         time_step = input.size(2)

#         step_sum = input.sum(1)  # B, T
#         step_pow_sum = input.pow(2).sum(1)  # B, T
#         #cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
#         #cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
#         cum_sum = tf.math.cumsum(step_sum, axis=1)  # B, T
#         cum_pow_sum = tf.math.cumsum(step_pow_sum, axis=1)  # B, T

#         entry_cnt = np.arange(channel, channel*(time_step+1), channel)
#         #entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
#         entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

#         cum_mean = cum_sum / entry_cnt  # B, T
#         cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
#         cum_std = (cum_var + self.eps).sqrt()  # B, T

#         cum_mean = cum_mean.unsqueeze(1)
#         cum_std = cum_std.unsqueeze(1)

#         x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
#         return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())































#hparams:
#hs:0.008
#loss_zero_ref_weight:1.0
#lr:0.0001
#lr_decay_rate:0.5
#lr_decay_steps:2000000
#mix_weights_type:'pred_source'
#num_sources_for_summaries:[1, 2, 3, 4]
#signal_names:['background', 'foreground_1', 'foreground_2', 'foreground_3']
#signal_types:['source', 'source', 'source', 'source']
#sr:16000.0
#ws:0.032

# signal.transformer init

# self.sample_rate = sample_rate
# self.magnitude_offset = magnitude_offset
# self.zeropad_beginning = zeropad_beginning

# # Compute derivative parameters.
# self.samples_per_window = int(round(sample_rate * window_time_seconds))
# self.hop_time_samples = int(round(self.sample_rate * hop_time_seconds))

# if num_basis <= 0:
#   self.fft_len = signal_util.enclosing_power_of_two(self.samples_per_window)
# else:
#   assert num_basis >= self.samples_per_window
#   self.fft_len = num_basis
# self.fft_bins = int(self.fft_len / 2 + 1)
  
def improved_tdcn(input_activations, config):
  """Creates improved time-dilated convolution network (TDCN++) from [1].

  [1] Ilya Kavalerov, Scott Wisdom, Hakan Erdogan, Brian Patton, Kevin Wilson,
      Jonathan Le Roux, John R. Hershey, "Universal Sound Separation,"
      https://arxiv.org/abs/1905.03330.
  Total number of convolutional layers is num_conv_blocks * num_repeats.

  Args:
    input_activations: Tensor (batch_size, num_frames, mics x depth, bins)
      of mixture input spectrograms.
    config: network_config.ImprovedTDCN object.
  Returns:
    layer_activations: activations of the last convolution of shape
        (batch_size, num_frames, bottleneck x mics x depth).
  """
  batch_size = signal_util.static_or_dynamic_dim_size(input_activations, 0)
  num_frames = signal_util.static_or_dynamic_dim_size(input_activations, 1)
  mics_and_depth = signal_util.static_or_dynamic_dim_size(input_activations, 2)

  layer_activations = input_activations
  # layer_activations is shape (batch_size, num_frames, mics x depth, bins).

  initial_dense_config = update_config_from_kwargs(
      config.initial_dense_layer,
      num_outputs=config.prototype_block[0].bottleneck)
  with tf.variable_scope('initial_dense'):
    layer_activations = dense_layer(layer_activations, initial_dense_config)

  input_of_block = []
  num_blocks = len(config.block_prototype_indices)
  find_scale_fn = _find_scale_function(config.scale_tdcn_block)
  with tf.variable_scope('improved_tdcn'):
    for block in range(num_blocks):
      proto_block = config.block_prototype_indices[block]
      connections_to_block = []
      for src, dest in zip(config.skip_residue_connection_from_input_of_block,
                           config.skip_residue_connection_to_input_of_block):
        if dest == block:
          assert src < dest
          connections_to_block.append(src)
      for prev_block in connections_to_block:
        residue_dense_config = update_config_from_kwargs(
            config.residue_dense_layer,
            num_outputs=config.prototype_block[proto_block].bottleneck)
        with tf.variable_scope('res_dense_{}_to_{}'.format(prev_block, block)):
          layer_activations += dense_layer(input_of_block[prev_block],
                                           residue_dense_config)
      input_of_block.append(layer_activations)
      scale_tdcn_block = find_scale_fn(block)
      # conv_inputs is shape
      # (batch_size, num_frames, mics x depth, bottleneck).

      with tf.variable_scope('conv_block_%d' % block):
        dilation = config.block_dilations[block]
        tdcn_block_config = config.prototype_block[proto_block]
        tdcn_block_config = update_config_from_kwargs(
            tdcn_block_config,
            scale=scale_tdcn_block,
            dilation=dilation)

        layer_activations = tdcn_block(
            layer_activations,
            tdcn_block_config)
    # layer_activations is of shape
    # (batch_size, num_frames, mics x depth, output_size).
    num_frames = signal_util.static_or_dynamic_dim_size(layer_activations, 1)
    # output_size = bottleneck when resid=true, concat_input=false.
    # output_size = num_blocks * bottleneck + num_coeff when
    #   resid=false, concat_input=true.
    output_size = signal_util.static_or_dynamic_dim_size(layer_activations, -1)
    layer_activations = tf.reshape(
        layer_activations,
        (batch_size, num_frames,
         mics_and_depth * output_size))
    # layer_activations is now shape
    # (batch_size, num_frames, mics x depth x bottleneck).
  return layer_activations

def get_activation_fn(name):
    """Returns an activation function."""
    act_fns = {
        'sigmoid': tf.nn.sigmoid,
        'relu': tf.nn.relu,
        'leaky_relu': tf.nn.leaky_relu,
        'tanh': tf.nn.tanh,
        'prelu': tfa.activations.PReLU,
        'linear': None,
    }
    if name not in act_fns:
        raise ValueError('Unsupported activation %s' % name)
    return act_fns[name]


def main():
    #tf.executing_eagerly()
    
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

    model_fn(params, encoder_params, separator_params, decoder_params)
    
    print('done')


if __name__ == "__main__":
    main()

