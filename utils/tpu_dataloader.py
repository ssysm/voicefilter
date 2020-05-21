import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import torch_xla.core.xla_model as xm
from utils.audio import Audio
import config


def create_dataloader(train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_mag_list = list()
        mixed_mag_list = list()

        for dvec_mel, target_mag, mixed_mag in batch:
            dvec_list.append(dvec_mel)
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        return dvec_list, target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        dataset = VFDataset(True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
        return DataLoader(dataset=dataset,
                          batch_size=config.train['batch_size'],
                          num_workers=config.train['num_workers'],
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=train_sampler)
    else:
        dataset = VFDataset(False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(dataset=dataset,
                          collate_fn=test_collate_fn,
                          sampler=test_sampler,
                          batch_size=1, shuffle=False, num_workers=1)


class VFDataset(Dataset):
    def __init__(self, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
        self.train = train
        self.data_dir = config.data['records_dir'] + config.data['train_dir'] if train else config.data['records_dir'] + config.data['test_dir']

        self.dvec_list = find_all(config.form['dvec'])
        self.target_wav_list = find_all(config.form['target']['wav'])
        self.mixed_wav_list = find_all(config.form['mixed']['wav'])
        self.target_mag_list = find_all(config.form['target']['mag'])
        self.mixed_mag_list = find_all(config.form['mixed']['mag'])

        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

        self.audio = Audio()

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()

        dvec_wav, _ = librosa.load(config.data['base_dir'] + dvec_path, sr=config.audio['sample_rate'])
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train: # need to be fast
            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])
            return dvec_mel, target_mag, mixed_mag
        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], config.audio['sample_rate'])
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], config.audio['sample_rate'])
            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        wav, _ = librosa.load(path, config.audio['sample_rate'])
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
