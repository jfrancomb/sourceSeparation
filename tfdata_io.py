import tensorflow as tf
import os 
import collections

# Read in wav files.
def decode_wav(wav, nsamples):
    audio_bytes = tf.io.read_file(wav)
    waveform, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1,
                                    desired_samples=nsamples)
    waveform = tf.reshape(waveform, (1, nsamples))
    return waveform

def decode_wav_or_return_zeros(wav, nsamples):
    return tf.cond(tf.equal(wav, '0'),
                lambda: tf.zeros((1, nsamples), dtype=tf.float32),
                lambda: decode_wav(wav, nsamples))

# Build mixture and sources waveforms.
def combine_mixture_and_sources(waveforms, nsamples, max_combined_sources):
    # waveforms is shape (max_combined_sources, 1, num_samples).
    mixture_waveform = tf.reduce_sum(waveforms, axis=0)
    source_waveforms = tf.reshape(waveforms,
                                (max_combined_sources, 1, nsamples))
    # reshaping here instead of in train_step
    mixture_waveform = tf.reshape(mixture_waveform, (mixture_waveform.shape[1],
                                                     mixture_waveform.shape[0]))
    source_waveforms = tf.squeeze(source_waveforms, 1)
    return (mixture_waveform, source_waveforms)

def read_lines_from_file(file_list_path, skip_fields=0, base_path='relative'):
    """Read lines from a file.
    Args:
      file_list_path: String specifying absolute path of a file list.
      skip_fields: Skip first n fields in each line of the file list.
      base_path: If `relative`, use base path of file list. If None, don't add
          any base path. If not None, use base_path to build absolute paths.

    Returns:
      List of strings, which are tab-delimited absolute file paths.
    """
    # Read in and preprocess the file list.
    with open(file_list_path, 'r') as f:
      lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split('\t')[skip_fields:] for line in lines]
    # Make each relative path point to an absolute path.
    lines_abs_path = []
    if base_path == 'relative':
      base_path = os.path.dirname(file_list_path)
    elif base_path is None:
      base_path = ''
    for line in lines:
      wavs_abs_path = []
      for wav in line:
        wavs_abs_path.append(os.path.join(base_path, wav))
      lines_abs_path.append(wavs_abs_path)
    lines = lines_abs_path
    # Rejoin the fields to return a list of strings.
    return ['\t'.join(fields) for fields in lines]

def unique_classes_from_lines(lines):
  """Return sorted list of unique classes that occur in all lines."""
  # Build sorted list of unique classes.
  unique_classes = sorted(
      list({x.split(':')[0] for line in lines for x in line}))  # pylint: disable=g-complex-comprehension
  return unique_classes

def wavs_to_dataset(file_list,
                        batch_size,
                        nsamples,
                        prefetch_buffer_size=1,
                        shuffle_data=False,
                        shuffle_buffer_size=50,
                        num_examples=-1,
                        repeat=True):
    r"""Fetches features from list of wav files.

    Args:
        file_list: List of tab-delimited file locations of wavs. Each line should
            correspond to one example, where each field is a source wav.
        batch_size: The number of examples to read.
        nsamples: Number of samples in each wav file.
        prefetch_buffer_size: Number of fetches that should happen in parallel.
        randomize_order: Whether to randomly shuffle features.
        nexamples: Limit number of examples to this value.  Unlimited if -1.
        shuffle_buffer_size: The size of the shuffle buffer.
        repeat: If True, repeat the dataset.

    Returns:
        A batch_size number of features constructed from wav files.
    """
    # need the skip_fields here set to one or else the mixture file is included 
    file_list = read_lines_from_file(file_list, skip_fields=1)
    lines = [line.split('\t') for line in file_list]
    max_component_sources = max([len(line) for line in lines])
    max_combined_sources = max_component_sources
    # Examples that have fewer than max_component_sources are padded with zeros.
    lines = [line + ['0'] * (max_component_sources - len(line)) for line in lines]
    wavs = []
    for line in lines:
        for wav in line:
            wavs.append(wav)
    wavs = tf.constant(wavs)
    dataset = tf.data.Dataset.from_tensor_slices(wavs)
    # takes input wavs from file and reads them into tensors. If zero elements,fills with 0 wavs
    dataset = dataset.map(lambda w: decode_wav_or_return_zeros(w, int(nsamples)))
    dataset = dataset.batch(max_component_sources)
    dataset = dataset.map(lambda w: combine_mixture_and_sources(w, nsamples, max_combined_sources))

    if shuffle_data:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    dataset = dataset.take(num_examples)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat:
        dataset = dataset.repeat()
    # changing the below line for v2 based on https://github.com/tensorflow/tensorflow/issues/29252
    #iterator = dataset.make_one_shot_iterator()
    # my change
    #iterator = iter(dataset)
    #return next(iterator)
    return dataset

def main_solo():
    # CMD python -m debugpy --listen 0.0.0.0:5678 --wait-for-client /home/jim/projects/sound-separation/models/dcase2020_fuss_baseline/train_model.py \
    # -dd=/home/jim/projects/sound-separation/datasets/fuss/data/fuss_dev/ssdata_reverb/ \
    # -md=/home/jim/projects/sound-separation/models/dcase2020_fuss_baseline/mytraining

    data_dir = '/home/jim/projects/sound-separation/datasets/fuss/data/fuss_dev/ssdata_reverb/'
    train_list = os.path.join(data_dir, 'train_example_list.txt')

    sr = 16000
    nsamples = int(sr * 10)
    prefetch_buffer_size = 512
    train_batch_size = 3

    dataset = wavs_to_dataset(file_list = train_list,
                                nsamples = nsamples,
                                batch_size = train_batch_size,
                                prefetch_buffer_size = prefetch_buffer_size,
                                shuffle_data = False,
                                num_examples = -1,
                                shuffle_buffer_size = 50,
                                repeat = False)


