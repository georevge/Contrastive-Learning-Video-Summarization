# -*- coding: utf-8 -*-
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
from tqdm import tqdm, trange
from layers.summarizer import PGL_SUM
from utils import TensorboardWriter


class Solver(object):
    def __init__(self, test_config=None, test_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        self.config = test_config
        self.test_loader = test_loader
        with open(self.config.save_dir.joinpath("best_model.pkl"), "rb") as file:
            self.model = pickle.load(file)

    def evaluate(self):
        self.model.eval()
        # epoch = str(epoch_i)
        # torch.save(self.model.state_dict(), self.config.save_dir.joinpath({epoch}.tar'))
        out_scores_dict = {}
        for frame_features, cap_features, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
            cap_features = cap_features.to(self.config.device)

            with torch.no_grad():
                video_embedding, scores = self.model(frame_features, cap_features)  # [1, seq_len]
                # scores = scores.squeeze(0).cpu().numpy().tolist()
                scores = scores.cpu().numpy().tolist()

                out_scores_dict[video_name] = scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            scores_save_path = self.config.score_dir.joinpath(f"{self.config.video_type}_test.json")
            with open(scores_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(scores_save_path)}.')
                json.dump(out_scores_dict, f)
            scores_save_path.chmod(0o777)




if __name__ == '__main__':
    pass
