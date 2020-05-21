import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
import pdb
import config

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % (num)))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(args, audio, num, s1_dvec, s1_target, s2, train):
    srate = config.audio['sample_rate']
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * config.embedder['window'] * config.audio['hop_length']:
        return

    # dataset have many silent interval, so let's vad-merge them
    if args.vad:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * config.data['audio_len'])
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # save vad & normalized wav files
    target_wav_path = formatter(dir_, config.form['target']['wav'], num)
    mixed_wav_path = formatter(dir_, config.form['mixed']['wav'], num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude spectrograms
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    target_mag_path = formatter(dir_, config.form['target']['mag'], num)
    mixed_mag_path = formatter(dir_, config.form['mixed']['mag'], num)
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(dir_, config.form['dvec'], num)
    with open(dvec_text_path, 'w') as f:
        s1_dvec = s1_dvec.split('/')
        if train == True :
            s1_dvec = s1_dvec[len(s1_dvec) - 4] + '/' + \
                        s1_dvec[len(s1_dvec) - 3] + '/' + s1_dvec[len(s1_dvec) - 2 ] + \
                        '/' + s1_dvec[len(s1_dvec) - 1]
        else:
            s1_dvec = s1_dvec[len(s1_dvec) - 3] + '/' + s1_dvec[len(s1_dvec) - 2 ] + \
                        '/' + s1_dvec[len(s1_dvec) - 1]
        f.write(s1_dvec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_dir', type=str, default=None,
                        help="Directory of VERDICT")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=bool, default=True, required=False,
                        help='apply vad(remove slience) to wav file. yes(true) or no(false, default)')
    args = parser.parse_args()

    output_path = args.out_dir
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if args.input_dir is None:
        raise Exception("Please provide directory of data")


    input_path = args.input_dir
    train_folders = [x for x in glob.glob(os.path.join(input_path, 'train', '*')) if os.path.isdir(x)]
    test_folders = [x for x in glob.glob(os.path.join(input_path, 'test', '*'))]

    train_spk = [glob.glob(os.path.join(spk, '**', config.form['input']), recursive=True)
                    for spk in train_folders]
    # train_spk = [x for x in train_folders if len(x) >= 2]
    test_spk = [glob.glob(os.path.join(spk, '**', config.form['input']), recursive=True)
                    for spk in test_folders]

    audio = Audio()

    def train_wrapper(num):
        spk1, spk2 = random.sample(train_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        mix(args, audio, num, s1_dvec, s1_target, s2, train=True)

    def test_wrapper(num):
        spk1, spk2 = random.sample(test_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        mix(args, audio, num, s1_dvec, s1_target, s2, train=False)

    arr = list(range(10**5))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    arr = list(range(10**3))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))

# python3 generator.py -c ./config.yaml -d /Users/renpeng/Downloads/vtb-sound-dataset -o /Users/renpeng/Downloads/voicefilter_output -p 6