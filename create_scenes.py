import os
import numpy as np
import pandas as pd
import shutil
import scaper

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

# def format_records(wav_dir = "datasets/scenes/", output_dir = "datasets/scenes/tfRecords"):
#   """converts the generated soundscapes to tensorflow records"""
#   files = os.listdir(wav_dir)
#   # just get file names without extensions
#   fnames = [f.split('.')[0] for f in files]
#   for fname in fnames:
#     jams_file = os.path.join(wav_dir, fname + ".jams")
#     wav_file = os.path.join(wav_dir, fname + ".wav")
#     jam = jams.load(jams_file)
#     with open(jams_file, "r") as read_file:
#       js = json.load(read_file)
#     sound_annotations = js['annotations'][0]
#     y = get_sound_response(wav_file, start_time, duration, mix_duration, sr)

if __name__ ==  "__main__":
    create_scaper_directory()
    scape_sounds()

