import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
import json

class VGGSound(Dataset):

    def __init__(self, config, mode='train'):
        self.config=config
        self.mode = mode
        self.use_pre_frame = 2
        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        with open("/data/wfq/paper/down/label_encoding.json", 'r') as f:
            class_class = json.load(f)

        if mode == "train":
            with open('/data/wfq/paper/down/train.csv') as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    audio_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/AudioWAV", item[2][1:] + ".wav")
                    video_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/Image-01-FPS", item[2][1:])
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        train_label.append(class_class[item[1]])

                self.video = train_video_data
                self.audio = train_audio_data
                self.label = train_label
        elif mode == "test":
            with open('/data/wfq/paper/down/test.csv') as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:

                    audio_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/AudioWAV", item[2][1:] + ".wav")
                    video_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/Image-01-FPS", item[2][1:])
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        test_label.append(class_class[item[1]])

            self.video = test_video_data
            self.audio = test_audio_data
            self.label = test_label

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=35400, mono=True)
        while len(sample) / rate < 20.:
            sample = np.tile(sample, 2)
        start_point = 0
        new_sample = sample[start_point:start_point + rate * 20]
        # new_sample = np.tile(sample, 20)[:22050 * 20]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.
        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.video[idx])
        if self.mode == 'train':
            select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
            select_index.sort()
        else:
            select_index = [idx for idx in range(0, len(image_samples), len(image_samples) // self.use_pre_frame)]
        images = torch.zeros((self.use_pre_frame, 3, 384, 384))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img
        images = images.permute((1, 0, 2, 3))
        one_hot = np.eye(self.config["setting"]["num_class"])
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)

        return spectrogram, images, label
    


class SemiVGGSound(Dataset):

    def __init__(self, config, mode='semi-label'):
        self.config=config
        self.mode = mode
        self.use_pre_frame = 2
        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        with open("/data/wfq/paper/down/label_encoding.json", 'r') as f:
            class_class = json.load(f)

        if mode == "test":
            with open('/data/wfq/paper/down/test.csv') as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:

                    audio_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/AudioWAV", item[2][1:] + ".wav")
                    video_dir = os.path.join("/data/wfq/paper/down/precessed_VGGSound/Image-01-FPS", item[2][1:])
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        test_label.append(class_class[item[1]])

            self.video = test_video_data
            self.audio = test_audio_data
            self.label = test_label

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=35400, mono=True)
        while len(sample) / rate < 20.:
            sample = np.tile(sample, 2)
        start_point = 0
        new_sample = sample[start_point:start_point + rate * 20]
        # new_sample = np.tile(sample, 20)[:22050 * 20]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.
        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.video[idx])
        if self.mode == 'train':
            select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
            select_index.sort()
        else:
            select_index = [idx for idx in range(0, len(image_samples), len(image_samples) // self.use_pre_frame)]
        images = torch.zeros((self.use_pre_frame, 3, 384, 384))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img
        images = images.permute((1, 0, 2, 3))
        one_hot = np.eye(self.config["setting"]["num_class"])
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)

        return spectrogram, images, label