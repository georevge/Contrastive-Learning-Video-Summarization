# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
from tqdm import tqdm, trange
from layers.summarizer import PGL_SUM
from utils import TensorboardWriter
import pickle


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Solver(object):
    def __init__(self, config=None, train_loader=None, validation_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        self.model = PGL_SUM(input_size=self.config.input_size,
                             output_size=self.config.input_size,
                             num_segments=self.config.n_segments,
                             heads=self.config.heads,
                             fusion=self.config.fusion,
                             pos_enc=self.config.pos_enc).to(self.config.device)
        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = Similarity(temp=0.1)
    # criterion = nn.MSELoss()
    loss_fct = nn.CrossEntropyLoss()  # reduction='none'

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - 0.5)

    def reconstruction_loss(self, video_embedding, attentive_ft):

        return torch.norm(video_embedding - attentive_ft, p=2)

    @staticmethod
    def repelling_loss(x1, x2):
        n = x1.size(0)  # number of samples (aka frames)

        # Normalize each -row- vector to l2_norm=1 (unit-norm)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)

        # Each frame compared with each frame, is basically the matrix multiplication w/o the main diagonal
        rep_loss = torch.matmul(x1, x2.t())
        rep_loss = rep_loss.fill_diagonal_(0.).sum().div(n * (n - 1))

        return rep_loss

    def variance_loss(self, scores, epsilon=1e-4):
        median_tensor = torch.zeros(scores.shape[1]).to("cuda")
        median_tensor.fill_(torch.median(scores))
        loss = nn.MSELoss()
        variance = loss(scores.squeeze(), median_tensor)
        return 1 / (variance + epsilon)

    def train(self):
        """ Main function to train the PGL-SUM model. """
        number_of_augm = 4
        if self.config.verbose:
            tqdm.write('Time to train the model...')

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()

            loss_history = []
            contrastive_loss_history = []
            repelling_loss_history = []
            sparsity_loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
                self.optimizer.zero_grad()
                embeddings = []
                mean_similarity_list = []
                h = []
                sparsity_losses = []
                repelling_losses = []
                for _ in trange(self.config.batch_size, desc='Video', ncols=80, leave=False):
                    sparsity_losses_per_video = []
                    repelling_losses_per_video = []

                    frame_features, cap_features = next(iterator)

                    frame_features = frame_features.to(self.config.device)
                    cap_features = cap_features.to(self.config.device)
                    for k in range(number_of_augm):
                        output, scores = self.model(frame_features.squeeze(0), cap_features.squeeze(0))
                        h.append(output)

                        sp_loss = self.sparsity_loss(scores)
                        sparsity_losses_per_video.append(sp_loss)
                        # rep_loss = self.repelling_loss(phi, phi)
                        # repelling_losses_per_video.append(rep_loss)

                    sparsity_loss_per_video = torch.stack(sparsity_losses_per_video)
                    # repelling_loss_per_video = torch.stack(repelling_losses_per_video)
                    sparsity_loss_per_video = torch.mean(sparsity_loss_per_video)
                    # repelling_loss_per_video = torch.mean(repelling_loss_per_video)
                    sparsity_losses.append(sparsity_loss_per_video)
                    # repelling_losses.append(repelling_loss_per_video)

                sparsity_loss = torch.stack(sparsity_losses)
                # repelling_loss = torch.stack(repelling_losses)
                sparsity_loss = (10/3) * torch.mean(sparsity_loss)
                # repelling_loss = 0.5 * torch.mean(repelling_loss)

                h_all = torch.stack(h, dim=0).squeeze(1)  #.detach().cpu().numpy()

                cos_sim_matrix = self.cos_sim(h_all.unsqueeze(1), h_all.unsqueeze(0)).unsqueeze(0)  #.detach().cpu().numpy()
                # cos_sim_matrix = torch.squeeze(cos_sim_matrix, dim=3)  # Only for text-visual
                # https://bhuvana-kundumani.medium.com/implementation-of-simcse-for-unsupervised-approach-in-pytorch-a3f8da756839
                cos_sim_matrix.squeeze(0).fill_diagonal_(0.)
                factor = number_of_augm / (number_of_augm-1)
                avg_pooling = nn.AvgPool2d(number_of_augm)
                cos_sim_matrix_avg = avg_pooling(cos_sim_matrix).squeeze(0)
                ones_mat = torch.ones(cos_sim_matrix_avg.shape).squeeze(0).fill_diagonal_(factor).to('cuda')
                cos_sim_matrix_avg = torch.mul(cos_sim_matrix_avg, ones_mat)
                labels = torch.arange(cos_sim_matrix_avg.size(0)).long().to("cuda")

                contrastive_loss = (1/10) * self.loss_fct(cos_sim_matrix_avg, labels)
                loss = contrastive_loss + sparsity_loss  # + repelling_loss

                # for i in range(self.config.batch_size):
                    # loss = losses[i]
                if self.config.verbose:
                    tqdm.write(f'[{epoch_i}] loss: {loss.item()}')
                loss.backward()
                loss_history.append(loss.data)
                # repelling_loss_history.append(repelling_loss.data)
                sparsity_loss_history.append(sparsity_loss.data)
                contrastive_loss_history.append(contrastive_loss.data)
                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()
            # repelling_loss = torch.stack(repelling_loss_history).mean()
            sparsity_loss = torch.stack(sparsity_loss_history).mean()
            contrastive_loss = torch.stack(contrastive_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')

            self.writer.update_loss(loss, epoch_i, 'total_loss_epoch')
            # self.writer.update_loss(repelling_loss, epoch_i, 'repelling_loss_epoch')
            self.writer.update_loss(sparsity_loss, epoch_i, 'sparsity_loss_epoch')
            self.writer.update_loss(contrastive_loss, epoch_i, 'contrastive_loss_epoch')

            # Uncomment to save parameters at checkpoint
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            pickle.dump(self.model, open(ckpt_path, 'wb'))

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i, save_weights=False):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        :param bool save_weights: Optionally, the user can choose to save the attention weights in a (large) h5 file.
        """
        self.model.eval()

        weights_save_path = self.config.score_dir.joinpath("weights.h5")
        out_scores_dict = {}
        for frame_features, cap_features, video_name in tqdm(self.validation_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
            # frame_features_cap = frame_features_cap.view(-1, self.config.input_size).to(self.config.device)
            cap_features = cap_features.to(self.config.device)

            with torch.no_grad():
                video_embedding, scores = self.model(frame_features, cap_features)  # [1, seq_len]
                # scores = scores.squeeze(0).cpu().numpy().tolist()
                scores = scores.squeeze(0).cpu().numpy().tolist()

                out_scores_dict[video_name] = scores

            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            scores_save_path = self.config.score_dir.joinpath(f"{self.config.video_type}_{epoch_i}.json")
            with open(scores_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(scores_save_path)}.')
                json.dump(out_scores_dict, f)
            scores_save_path.chmod(0o777)

            if save_weights:
                with h5py.File(weights_save_path, 'a') as weights:
                    weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=scores)


if __name__ == '__main__':
    pass
