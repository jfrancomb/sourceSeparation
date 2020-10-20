import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import cv2
import shutil
import scaper

def scale_minmax(X, min=0.0, max=1.0):
    """takes an image as np array and scales it between provided min and max"""
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def alpha_blend(a, b, alpha):
    """returns weighted blend of images a and b based on the provided weight"""
    return a * alpha + (1-alpha) * b

def gen_noise(shape):
    """returns gaussian noise with given shape"""
    return np.random.standard_normal(shape)

def get_cqt_spectrogram(y, sr = 22050, fmin = 32.7, n_bins = 72, hop_length = 512):
    """generates a constant q transformed spectrogram"""
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    return img

def get_chroma_cqt(y, sr = 22050):
    """generates a constant q chromagram"""
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    return chroma_cq

def generate_spectrogram_directory(wav_dir, spec_dir, sr = 22050, fmin = 32.7, n_bins = 72, hop_length = 512):
    """takes as input the path to a directory containing the wav dataset, converts all files to spectrograms and writes them to the spec_dir""" 
    wav_files = os.listdir(wav_dir)
    wav_files = [wav_dir + f if '.wav' in f else None for f in wav_files]
    for wav in wav_files:
        # read wav
        y, sr = librosa.load(wav,  sr = sr)
        # get and plot spectrogram
        img = get_cqt_spectrogram(y, sr, fmin = fmin, n_bins = n_bins, hop_length = hop_length)
        # save  image
        png = wav.replace('wav', 'png').replace('audio','img')
        cv2.imwrite(png, img)

def get_esc50_classification_splits(img_dir = 'ESC-50-master/img/', class_labels = 'ESC-50-master/meta/esc50.csv', train_folds = [1,2,3,4]):
    """Takes specgrogram image input directory and retrieves ESC50 train and test sets based on specified folds in the class_labels csv file"""
    label_df = pd.read_csv(class_labels)
    img_files = os.listdir(img_dir)
    train_img = []
    train_label = []
    test_img = []
    test_label = []
    class_map = {}
    for img_file in img_files:
        img_path = img_dir + img_file
        img = cv2.imread(img_path)
        #get fold info
        img_row = label_df[label_df['filename'] == img_file.replace('png','wav')]
        target = img_row.iloc[0]['target']
        if int(img_row['fold']) in train_folds:
            train_img.append(img)
            train_label.append(target)
        else:
            test_img.append(img)
            test_label.append(target)
        if target not in class_map.keys():
            class_map[target] = img_row.iloc[0]['category']
    train_img = np.array(train_img)
    train_label = np.array(train_label)
    test_img = np.array(test_img)
    test_label = np.array(test_label)
    return train_img, train_label, test_img, test_label, class_map

def create_scaper_directory(input_dir = 'datasets/ESC-50-master/audio',
                            output_dir = 'datasets/scaper/', 
                            class_labels = 'datasets/ESC-50-master/meta/esc50.csv',
                            foreground_classes = ['cat','dog','laughing','can_opening','sheep','hen','clapping','crow','glass_breaking'],
                            background_classes = ['vacuum_cleaner','thunderstorm','chainsaw','airplane','train', 'washing_machine']):
    # determine whether this class is foreground, background or neither
    label_df = pd.read_csv(class_labels)
    wav_files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for wav_file in wav_files:
        # get the sound's class
        wav_row = label_df[label_df['filename'] == wav_file]
        wav_class = wav_row.iloc[0]['category']
        # Move to foreground directory 
        if wav_class in foreground_classes:
            infile = os.path.join(input_dir, wav_file)
            outfile = os.path.join(output_dir + '/foreground/' + wav_class + '/', wav_file)
            if not os.path.exists(output_dir + '/foreground/' + wav_class):
                os.makedirs(output_dir + '/foreground/' + wav_class + '/')
            shutil.copyfile(src = infile, dst = outfile)
        # Move to background directory
        elif wav_class in background_classes:
            infile = os.path.join(input_dir, wav_file)
            outfile = os.path.join(output_dir + '/background/' + wav_class + '/', wav_file)
            if not os.path.exists(output_dir + '/background/' + wav_class):
                os.makedirs(output_dir + '/background/' + wav_class + '/')
            shutil.copyfile(src = infile, dst = outfile)    
    return 0

def scape_sounds(out_dir = 'datasets/scenes/',
                fg_dir = 'datasets/scaper/foreground/', 
                bg_dir = 'datasets/scaper/background',
                foreground_classes = ['cat','dog','laughing','can_opening','sheep','hen','clapping','crow','glass_breaking'],
                background_classes = ['vacuum_cleaner','thunderstorm','chainsaw','airplane','train', 'washing_machine'],
                n_soundscapes = 1000, ref_db = -50, duration = 10.0, min_events = 1, max_events = 9, 
                event_time_dist = 'truncnorm', event_time_mean = 5.0, event_time_std = 2.0, event_time_min = 0.0,
                event_time_max = 10.0, source_time_dist = 'const', source_time = 0.0, event_duration_dist = 'uniform', 
                event_duration_min = 0.5, event_duration_max = 4.0, snr_dist = 'uniform', snr_min = 6, snr_max = 30, 
                pitch_dist = 'uniform', pitch_min = -3.0, pitch_max = 3.0, time_stretch_dist = 'uniform',
                time_stretch_min = 0.8, time_stretch_max = 1.2):
    for n in range(n_soundscapes):
        print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
        # creating a scaper object 
        sc = scaper.Scaper(duration, fg_dir, bg_dir)
        sc.protected_labels = []
        sc.ref_db = ref_db
        # addding background
        sc.add_background(label=('choose', background_classes), 
                        source_file=('choose', []), 
                        source_time=('const', 0))
        # add random number of foreground events
        n_events = np.random.randint(min_events, max_events+1)
        for _ in range(n_events):
            sc.add_event(label=('choose', foreground_classes), 
                        source_file=('choose', []), 
                        source_time=(source_time_dist, source_time), 
                        event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
                        event_duration=(event_duration_dist, event_duration_min, event_duration_max), 
                        snr=(snr_dist, snr_min, snr_max),
                        pitch_shift=(pitch_dist, pitch_min, pitch_max),
                        time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        # generate scapes
        audiofile = os.path.join(out_dir, "soundscape_unimodal{:d}.wav".format(n))
        jamsfile = os.path.join(out_dir, "soundscape_unimodal{:d}.jams".format(n))
        txtfile = os.path.join(out_dir, "soundscape_unimodal{:d}.txt".format(n))

        sc.generate(audiofile, jamsfile,
                    allow_repeated_label=True,
                    allow_repeated_source=False,
                    reverb=0.1,
                    disable_sox_warnings=True,
                    no_audio=False,
                    txt_path=txtfile)
    return 0

if __name__ ==  "__main__":
    create_scaper_directory()
    scape_sounds()

