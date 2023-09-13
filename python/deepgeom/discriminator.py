# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_r, n_c):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_r, n_c, 1)  # .cuda()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, code, h_pl, h_mi, s_bias1=None, s_bias2=None):

        sc_1 = self.f_k(h_pl.contiguous(), code.contiguous())
        sc_2 = self.f_k(h_mi.contiguous(), code.contiguous())
        logits = torch.cat((sc_1, sc_2), 1)

        return logits
