# -*- coding: utf-8 -*-

# dataloader block level

import torch
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
        self.datasets = [...,
                         ...,
                         ...,
                         ...]
        self.datasets_cap = [...,
                             ...]
        self.splits_filename = ['.../splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
            self.filename_cap = self.datasets_cap[0]
            self.filename_gt = self.datasets[2]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
            self.filename_cap = self.datasets_cap[1]
            self.filename_gt = self.datasets[3]
        hdf = h5py.File(self.filename, 'r')
        hdf_cap = h5py.File(self.filename_cap, 'r')
        hdf_gt = h5py.File(self.filename_gt, 'r')
        self.list_frame_features, self.list_frame_features_cap, self.list_gtscores = [], [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            shots_paths = hdf[video_name + '/features'].keys()
            n = len(shots_paths)
            video_level_ft_list = []
            for num_of_shot in range(1, n+1):
                blocks_paths = hdf[video_name + '/features' + '/shot_' + str(num_of_shot)]
                m = len(blocks_paths)
                shot_level_ft_list = []
                for num_of_block in range(1, m+1):
                    block_feature = torch.Tensor(np.array(hdf[video_name + '/features' + '/shot_' + str(num_of_shot) + '/block_' + str(num_of_block)]))
                    shot_level_ft_list.append(block_feature)

                shot_level_ft = torch.stack(shot_level_ft_list, dim=0)
                shot_level_ft = torch.mean(shot_level_ft, dim=0)
                video_level_ft_list.append(shot_level_ft)

            video_level_ft = torch.stack(video_level_ft_list, dim=0)
            video_level_ft = torch.squeeze(video_level_ft)

            frame_features_cap = torch.Tensor(np.array(hdf_cap[video_name + '/features']))

            gtscore = np.array(hdf_gt[video_name + '/gtscore'])

            change_points = np.array(hdf_gt[video_name + '/change_points'])
            n_frame_per_seg = np.array(hdf_gt[video_name + '/n_frame_per_seg'])
            picks = np.array(hdf_gt[video_name + '/picks'])

            video_level_gt_list = []
            for i in range(len(change_points)):
                left = change_points[i, 0]
                right = change_points[i, 1]
                n_frame = n_frame_per_seg[i]
                gtscore_per_shot_list = []
                while left % 15 != 0:
                    left = left + 1
                for j in range(left, right + 1, 15):
                    k = j // 15
                    gtscore_per_shot_list.append(gtscore[k])

                if not gtscore_per_shot_list:
                    gtscore_per_shot_list.append(0)

                gtscore_per_shot = torch.Tensor(gtscore_per_shot_list)
                gtscore_per_shot = torch.mean(gtscore_per_shot, dim=0)
                video_level_gt_list.append(gtscore_per_shot)

            video_level_gt_per_shot = torch.stack(video_level_gt_list, dim=0)
            # video_level_gt_per_shot = torch.squeeze(video_level_gt_per_shot)

            self.list_frame_features.append(video_level_ft)
            self.list_frame_features_cap.append(frame_features_cap)
            self.list_gtscores.append(video_level_gt_per_shot)

        hdf.close()
        hdf_cap.close()
        hdf_gt.close()

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
        frame_features_cap = self.list_frame_features_cap[index]

        if self.mode == 'test':
            video_name = self.split[self.mode + '_keys'][index]
            return frame_features, frame_features_cap, video_name
        else:
            gtscore = self.list_gtscores[index]
            return frame_features, frame_features_cap, gtscore


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
