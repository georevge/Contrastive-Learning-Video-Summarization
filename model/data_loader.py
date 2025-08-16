# -*- coding: utf-8 -*-
# shot level
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.initial_datasets = [...,
                                ...]
        self.datasets_cap = [..., ...]
        #self.extended_datasets = ['..., ...]
        self.splits_filename = ['/.../splits (training+val+test)/' + self.name + '_splits.json']
        # self.train_keys_filename = ['.../splits/' + self.name + '_train_keys.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if self.name == 'summe':
            # self.extended_filename = self.extended_datasets[0]
            self.initial_filename = self.initial_datasets[0]
            self.initial_filename_cap = self.datasets_cap[0]
        elif self.name == 'tvsum':
            # self.extended_filename = self.extended_datasets[1]
            self.initial_filename = self.initial_datasets[1]
            self.initial_filename_cap = self.datasets_cap[1]

        hdf_initial = h5py.File(self.initial_filename, 'r')
        # hdf_extended = h5py.File(self.extended_filename, 'r')
        hdf_cap = h5py.File(self.initial_filename_cap, 'r')

        self.list_frame_features = []
        self.list_frame_features_cap = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break
        for video_name in self.split[self.mode + '_keys']:
            video_features = torch.Tensor(np.array(hdf_initial[video_name + '/features']))
            num_sampled_frames = len(video_features)
            '''
            num_of_frgms = math.ceil(num_sampled_frames/4)
            list_ft = []
            for i in range(num_of_frgms):
                start = i * 4
                end = (i + 1) * 4
                if i == num_of_frgms - 1:
                    end = num_sampled_frames
                ft = video_features[start:end]
                fragment = torch.mean(ft, dim=0)
                list_ft.append(fragment)
            fragment_level = torch.stack(list_ft)
            self.list_frame_features.append(fragment_level)
            '''
            frame_features_cap = torch.Tensor(np.array(hdf_cap[video_name + '/features']))
            self.list_frame_features_cap.append(frame_features_cap)

            self.list_frame_features.append(video_features)

        hdf_initial.close()
        # hdf_extended.close()
        hdf_cap.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """

        frame_features = self.list_frame_features[index]
        text_features = self.list_frame_features_cap[index]
        if self.mode == 'validation':
            video_name = self.split[self.mode + '_keys'][index]
            return frame_features, text_features, video_name
        if self.mode == 'test':
            video_name = self.split[self.mode + '_keys'][index]
            return frame_features, text_features, video_name
        else:
            return frame_features, text_features


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)

if __name__ == '__main__':
    pass
