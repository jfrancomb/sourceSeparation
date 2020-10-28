
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import librosa
import jams

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def format_records(wav_dir = "datasets/scenes/", output_dir = "datasets/scenes/tfRecords"):
  """converts the generated soundscapes to tensorflow records"""
  files = os.listdir(wav_dir)
  # just get file names without extensions
  fnames = [f.split('.')[0] for f in files]
  for fname in fnames:
    jams_file = os.path.join(wav_dir, fname + ".jams")
    wav_file = os.path.join(wav_dir, fname + ".wav")
    jam = jams.load(jams_file)
    with open(jams_file, "r") as read_file:
      js = json.load(read_file)
    sound_annotations = js['annotations'][0]
    y = get_sound_response(wav_file, start_time, duration, mix_duration, sr)

def get_sound_response(wav_file, start_time, duration, mix_duration):
  np.array()
    
  print(wav_files)
  return 0

def format_esc50_records(wav_dir, metadata_file = 'datasets/ESC-50-master/meta/esc50.csv', output_dir = 'datasets/ESC-50-master/', copy = False):
    """reads all wav files in a directory, converts to tf examples and writes them to a single tfrecord file"""
    #init stuff before loop before loop
    note_id = 0
    instrument_name_id = 0
    instrument_names = {}
    wav_files = os.listdir(wav_dir)
    #proto_list = []
    record_file = os.path.join(output_dir, 'esc50.tfrecord')
    with tf.io.TFRecordWriter(record_file) as writer:
        for wav_file in wav_files:
            md = pd.read_csv(metadata_file)
            note_row = md[md.filename == wav_file]
            # tf record needs name of wav file
            instrument_str = wav_file.replace('.wav','')
            #instrument string components
            inst_name = str(note_row.iloc[0][3])
            inst_name = inst_name.replace('_','')
            if inst_name in instrument_names.keys():
                instrument_names[inst_name] += 1
                inst_num = instrument_names[inst_name]
            else:
                instrument_names[inst_name] = 1
            # audio stuff
            sample_rate = 16000
            audio = librosa.core.load(os.path.join(wav_dir,wav_file), sr = sample_rate)[0] # its a tuple with sample rate
            # instrument stuff 
            # instrument id and the third part of instrument_str are not the same in nsynth
            # I am doing that here - it shouldn't matter.
            instrument = instrument_names[inst_name] # unique id for each inst name
            instrument_str = inst_name+'_'+'acoustic'+'_'+str(instrument)
            instrument_source = 0
            instrument_source_str = 'acoustic' # using all acoustic - somewhat true
            instrument_family =  instrument # making same as instrument.. uni
            instrument_family_str = inst_name
            #note stuff
            pitch = 64
            velocity = 64
            note_str = instrument_str + '-' + str(pitch) + '-' + str(velocity)
            note = note_id
            qualities_str = np.array([b'multiphonic']) # using multiphonic because it's probably actually true
            qualities = np.array([0,0,0,0,0,0,0,0,0,0])
            ###### Starting actual record #######
            # preserves order from examples.json
            record = {
                'qualities'                 : tf.train.Feature(int64_list=tf.train.Int64List(value = qualities))
                ,'pitch'                    : _int64_feature(pitch)
                ,'note'                     : _int64_feature(note)
                ,'instrument_source_str'    : _bytes_feature(tf.compat.as_bytes(instrument_source_str))
                ,'velocity'                 : _int64_feature(velocity)
                ,'instrument_str'           : _bytes_feature(tf.compat.as_bytes(instrument_str))
                ,'instrument'               : _int64_feature(instrument)
                ,'sample_rate'              : _int64_feature(sample_rate)
                ,'qualities_str'            : tf.train.Feature(bytes_list = tf.train.BytesList(value = qualities_str))
                ,'instrument_source'        : _int64_feature(instrument_source)
                ,'note_str'                 : _bytes_feature(value = tf.compat.as_bytes(note_str))
                ,'audio'                    : tf.train.Feature(float_list = tf.train.FloatList(value = audio))
                ,'instrument_family'        : _int64_feature(instrument_family)
                ,'instrument_family_str'    : _bytes_feature(value = tf.compat.as_bytes(instrument_family_str))
            }
            proto = tf.train.Example(features=tf.train.Features(feature=record))
            proto = proto.SerializeToString()
            #proto_list.append(proto)
            writer.write(proto)
            print(note_str)
            #print(len(audio) / sample_rate) #all are 5 seconds
            if copy:
                #move wav file to new dir
                new_file = os.path.join(output_dir, note_str)
                shutil.copyfile(wav_file, new_file)
            note_id += 1
            
def main():
  #format_esc50_records(wav_dir='datasets/ESC-50-master/audio/',  output_dir= 'datasets/ESC-50-master')
  format_records()

if __name__ == "__main__":
  main()
