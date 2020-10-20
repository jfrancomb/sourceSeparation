import numpy as np
from create_scenes import *
from matplotlib import
import cv2
import librosa

if __name__ == "__main__":
    sr = 22050
    hop_length = 512
    n_bins = 72
    fmin = 32.7

    wav = 'datasets/ESC-50-master/audio/1-118206-A-31.wav'
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    png_fname = wav.replace('wav', 'png').replace('audio','img')# this is specific to ESC
    cv2.imwrite('mouse.png', img)

    noise = np.random.standard_normal(img.shape)
    noise = np.floor(scale_minmax(noise) * 255)
    noisy = alpha_blend(img, noise, .5)
    cv2.imwrite('noisymouse.png', noisy)

    wav = 'datasets/ESC-50-master/audio/1-118206-A-31.wav'
    y, sr = librosa.load(wav,  sr = sr)
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    png_fname = wav.replace('wav', 'png').replace('audio','img')# this is specific to ESC
    cv2.imwrite('mouse.png', img)

    noise = np.random.standard_normal(img.shape)
    noise = np.floor(scale_minmax(noise) * 255)
    noisy = alpha_blend(img, noise, .5)
    cv2.imwrite('noisymouse.png', noisy)



    wav = 'datasets/ESC-50-master/audio/1-115545-A-48.wav'
    y, sr = librosa.load(wav,  sr = sr)
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    png_fname = wav.replace('wav', 'png').replace('audio','img')# this is specific to ESC
    cv2.imwrite('fireworks.png', img)

    noise = np.random.standard_normal(img.shape)
    noise = np.floor(scale_minmax(noise) * 255)
    noisy = alpha_blend(img, noise, .5)
    cv2.imwrite('noisyfireworks.png', noisy)



    wav = 'datasets/ESC-50-master/audio/3-181132-A-14.wav'
    y, sr = librosa.load(wav,  sr = sr)
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    png_fname = wav.replace('wav', 'png').replace('audio','img')# this is specific to ESC
    cv2.imwrite('birdchirp.png', img)

    noise = np.random.standard_normal(img.shape)
    noise = np.floor(scale_minmax(noise) * 255)
    noisy = alpha_blend(img, noise, .5)
    cv2.imwrite('noisybirdchirp.png', noisy)

    wav = 'datasets/ESC-50-master/audio/1-118559-A-17.wav'
    y, sr = librosa.load(wav,  sr = sr)
    C = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    logC = librosa.amplitude_to_db(np.abs(C))
    img = scale_minmax(logC, 0, 255).astype(np.uint8)
    png_fname = wav.replace('wav', 'png').replace('audio','img')# this is specific to ESC
    cv2.imwrite('waterpour.png', img)

    noise = np.random.standard_normal(img.shape)
    noise = np.floor(scale_minmax(noise) * 255)
    noisy = alpha_blend(img, noise, .5)
    cv2.imwrite('noisywaterpour.png', noisy)
    img1 = img
    img2 = img
    cv2.imwrite('img1.png', img1)
    cv2.imwrite('img2.png', img2)

    blend1 = alpha_blend(img1,img2, alpha = .25)
    cv2.imwrite('blend25.png', blend1)

    blend2 = alpha_blend(img1,img2, alpha = .5)
    cv2.imwrite('blend50.png', blend2)

    blend3 = alpha_blend(img1,img2, alpha = .75)
    cv2.imwrite('blend75.png', blend3)

    gen_noise(img1.shape)



