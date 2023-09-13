# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# functions for foldingnet plus

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from deepgeom.discriminator import Discriminator
from deepgeom.pointnet import (
    GlobalPoolAndFC,
    PointNetAllLayers,
    PointNetGlobalMax,
    PointNetVanilla,
    PointwiseMLP,
    get_MLP_layers,
)
from scipy.fftpack import dct
from sklearn import manifold
from sklearn.utils import check_random_state

# In order to use NNDModule from a "C" implementation
# ./nndistance put side by side of ./python
sys.path.append("../nndistance/")
from modules.nnd import NNDModule


class ChamfersDistance(nn.Module):
    """
    Use NNDModule as a member.
    """

    def __init__(self):
        super(ChamfersDistance, self).__init__()
        self.nnd = NNDModule()

    def forward(self, input1, input2):  # BxNxK, BxMxK
        dist0, dist1 = self.nnd.forward(input1, input2)  # BxN, BxM
        # loss = torch.mean(torch.sqrt(dist0), 1) + torch.mean(torch.sqrt(dist1), 1)
        # minimize the max of cd
        loss = torch.max(torch.mean(torch.sqrt(dist0), 1), torch.mean(torch.sqrt(dist1), 1))
        loss = torch.mean(loss)  # 1
        return loss


class ChamfersDistance2(NNDModule):
    """
    Derive a new class from NNDModule
    """

    def forward(self, input1, input2):  # BxNxK, BxMxK
        dist0, dist1 = super(ChamfersDistance3, self).forward(input1, input2)  # BxN, BxM
        loss = torch.mean(torch.sqrt(dist0), 1) + torch.mean(torch.sqrt(dist1), 1)  # B
        loss = torch.mean(loss)  # 1
        return loss


class ChamfersDistance3(nn.Module):
    """
    Extensively search to compute the Chamfersdistance. No reference to external implementation Incomplete
    """

    def forward(self, input1, input2):
        B, N, K = input1.shape
        _, M, _ = input2.shape
        input11 = input1.unsqueeze(2)  # BxNx1xK
        input11 = input11.expand(B, N, M, K)  # BxNxMxK
        input22 = input2.unsqueeze(1)  # Bx1xMxK
        input22 = input22.expand(B, N, M, K)  # BxNxMxK
        D = input11 - input22  # BxNxMxK
        D = torch.norm(D, p=2, dim=3)  # BxNxM
        dist0, _ = torch.min(D, dim=1)  # BxM
        dist1, _ = torch.min(D, dim=2)  # BxN
        # augmented Chamfer distance
        loss = torch.max(torch.mean(torch.sqrt(dist0), 1), torch.mean(torch.sqrt(dist1), 1))
        loss = torch.mean(loss)  # 1
        return loss


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class FoldingNetVanilla(nn.Module):  # PointNetVanilla or nn.Sequential
    def __init__(
        self,
        MLP_dims,
        FC_dims,
        grid_dims,
        Folding1_dims,
        Folding2_dims,
        Weight1_dims,
        Weight3_dims,
        knn=96,
        sigma=0.008,
        MLP_doLastRelu=False,
    ):
        assert MLP_dims[-1] == FC_dims[0]
        super(FoldingNetVanilla, self).__init__()
        # Encoder
        self.PointNet = PointNetVanilla(MLP_dims, FC_dims, MLP_doLastRelu)

        # Decoder
        # create a 2D grid
        self.N = grid_dims[0] * grid_dims[1]
        u = (torch.arange(0.0, grid_dims[0] * 1.0) / grid_dims[0] - 0.5).repeat(grid_dims[1])
        v = (torch.arange(0.0, grid_dims[1] * 1.0) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
        self.grid = torch.stack((u, v), 1)  # Nx2
        u_matrix = torch.t(torch.unsqueeze(u, 0))
        v_matrix = torch.t(torch.unsqueeze(v, 0))
        grid_pc = torch.cat((u_matrix, v_matrix), 1)
        # initialize a weight matrix from the created 2D grid
        grid_pc_norm = torch.t(torch.unsqueeze((grid_pc**2).sum(1), 0))
        grid_pc_norm_mat1 = grid_pc_norm.expand(-1, grid_dims[0] * grid_dims[1])
        grid_pc_norm_mat2 = torch.t(grid_pc_norm).expand(grid_dims[1] * grid_dims[0], -1)
        dist_mat = grid_pc_norm_mat1 + grid_pc_norm_mat2 - 2 * torch.mm(grid_pc, torch.t(grid_pc))
        weight_mat = torch.exp(-dist_mat.float() * (1 / sigma) * torch.sqrt(torch.mean(dist_mat.float())))
        maxk_element, maxk_index = torch.topk(weight_mat, knn, 1)
        maxk_element[:, 0] = 0
        weight_mat_sumrow = torch.unsqueeze(torch.sum(maxk_element, 1), 1).expand(-1, knn)
        weight_normalize = maxk_element / weight_mat_sumrow  # normalize or not
        weight_initial = torch.zeros(2025, 2025)
        weight_normalize = weight_initial.scatter(1, maxk_index, weight_normalize)
        # check the max and min values of the weight matrix
        # print("max value of weight matrix:" , torch.max(weight_normalize))
        # print("min value of weight matrix:" , torch.min(weight_normalize))
        self.weight_mat_normalize = weight_normalize  # use original weight matrix as a filter
        #   1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #   2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)

        #   1st estimation
        self.Weight_estimate1 = FoldingNetSingle(Weight1_dims)
        #   2nd estimation
        self.Weight_estimate3 = FoldingNetSingle(Weight3_dims)

    def forward(self, X):
        # encoding
        f = self.PointNet.forward(X)  # BxK
        cw_1d = f
        f = f.unsqueeze(1)  # Bx1xK
        codeword = f.expand(-1, self.N, -1)  # BxNxK
        # cat 2d grid and feature
        B = codeword.shape[0]  # extract batch size
        B_train, N, _ = X.shape
        if not X.is_cuda:
            tmpGrid = self.grid  # Nx2
        else:
            tmpGrid = self.grid.cuda()  # Nx2
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)  # BxNx2
        num_m = tmpGrid.shape[1]
        if not X.is_cuda:
            weight_matrix = self.weight_mat_normalize

        else:
            weight_matrix = self.weight_mat_normalize.cuda()  # M x M
        weight_matrix = weight_matrix.unsqueeze(0)
        weight_matrix = weight_matrix.expand(B, -1, -1)  # BxMxM
        # 1st estimation for weight matrix
        w = torch.cat((weight_matrix, codeword), 2)  # Bx2025x(2025+512)
        w = self.Weight_estimate1.forward(w)  # Bx2025x2025
        # 2nd estimation for weight matrix
        w = torch.cat((w, codeword), 2)
        w = self.Weight_estimate3.forward(w)  # BxNxN
        # 0 here can be replaced by other values to make the filter stronger or weaker
        w[w < 0] = 0
        w_sum = torch.unsqueeze(torch.sum(w, 2), 2).expand(-1, -1, w.shape[2])
        # normalize the matrix to make the rows sum to one
        w = w / w_sum
        # create a symmetric weight matrix
        w = 0.5 * w + 0.5 * torch.transpose(w, 1, 2)

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2)  # BxNx(K+2)
        f = self.Fold1.forward(f)  # BxNx3 # B x N x 2 for isomap
        # 2nd folding
        f = torch.cat((f, codeword), 2)  # BxNx(K+3)
        f = self.Fold2.forward(f)  # BxNx3
        # f is the coarse version of reconstructed point clouds, w is the calculated graph/weight matrix, cw_1d is a 1d codeword for the corresponding point cloud
        return f, w, cw_1d
