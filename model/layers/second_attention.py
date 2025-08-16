# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize



class Attention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, cap_size=512):    # roberta=768
        """ The basic (multi-head) Attention 'cell' containing the learnable parameters of Q, K and V

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int freq: The frequency of the sinusoidal positional encoding.
        :param int heads: Number of heads for the attention module.
        :param str | None pos_enc: The type of the positional encoding [supported: Absolute, Relative].
        """
        super(Attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.Wk = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.Wq = nn.Linear(in_features=input_size, out_features=output_size, bias=False)  # for text: in_features=cap_size
        self.Wv = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.out = nn.Linear(in_features=output_size, out_features=768, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x, y):

        """ Compute the weighted frame features, based on either the global or local (multi-head) attention mechanism.

        :param torch.tensor x: Frame features with shape [T, input_size]
        :return: A tuple of:
                    y: Weighted features based on the attention weights, with shape [T, input_size]
                    att_weights : The attention weights (before dropout), with shape [T, T]
        """

        # Only visual

        x = self.drop(x)
        y = self.drop(y)
        y_avg = torch.mean(y, dim=0)
        K = self.Wk(x)
        Q = self.Wq(y_avg)
        V = self.Wv(x)
        '''
        # Text + video
        x = self.drop(x)
        K = self.Wk(x)
        Q = self.Wq(y)
        V = self.Wv(x)
        '''
        energies = torch.matmul(Q, K.transpose(1, 0))

        att_weights = self.softmax(energies)
        # _att_weights = self.drop(att_weights)
        output = torch.matmul(att_weights, V)  # _att_weights
        output = torch.unsqueeze(output, dim=0)
        # y = self.out(output)
        y = self.drop(output)
        return y, att_weights.clone()  # for now we don't deal with the weights (probably max or avg pooling)


if __name__ == '__main__':
    pass
    """Uncomment for a quick proof of concept
    model = SelfAttention(input_size=256, output_size=256, pos_enc="absolute").cuda()
    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
    """
