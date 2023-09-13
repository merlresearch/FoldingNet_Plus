# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch

# for modelnet40 dataset
seg_data_test = np.load("../modelNet40_test_2048.npy", encoding="latin1").item()
seg_data_data = seg_data_test["data"]
seg_data_label = seg_data_test["label"]
# for modelNet10 dataset
# seg_data_test = np.load("../modelNet10_test_2048.npy").item()
# seg_data_data = seg_data_test[b'data']
# seg_data_label = seg_data_test[b'label']
num_pc = seg_data_data.shape[0]
sigma = 0.008
knn_vector = [20, 40, 60]
knn_index = 0
while knn_index <= len(knn_vector) - 1:
    knn = knn_vector[knn_index]
    covariance_mat = []
    weight_covariance_mat = []

    for ii in range(num_pc):
        if ii % 500 == 0:
            print(ii)
        data_ii = np.squeeze(seg_data_data[ii, :, :])
        data_xyz_ii = data_ii[:, 0:3]
        data_xyz_tensor_ii = torch.from_numpy(data_xyz_ii)
        grid_pc_norm = torch.t(torch.unsqueeze((data_xyz_tensor_ii**2).sum(1), 0))
        grid_pc_norm_mat1 = grid_pc_norm.expand(-1, 2048)
        grid_pc_norm_mat2 = torch.t(grid_pc_norm).expand(2048, -1)
        # print(data_xyz_tensor_ii.shape)
        dist_mat = grid_pc_norm_mat1 + grid_pc_norm_mat2 - 2 * torch.mm(data_xyz_tensor_ii, torch.t(data_xyz_tensor_ii))
        weight_mat = torch.exp(-dist_mat.float() * (1 / sigma) * torch.sqrt(torch.mean(dist_mat.float())))
        maxk_element, maxk_index = torch.topk(weight_mat, knn, 1)
        maxk_element[:, 0] = 0
        weight_mat_sumrow = torch.unsqueeze(torch.sum(maxk_element, 1), 1).expand(-1, knn)
        weight_normalize = maxk_element / weight_mat_sumrow  # normalize or not
        weight_normalize_unsqueeze = torch.unsqueeze(weight_normalize, 0)
        maxk_index_unsqueeze = torch.unsqueeze(maxk_index, 0)
        data_neighbor_ii = data_xyz_tensor_ii[maxk_index, :]  # 2048 x knn x 3
        data_neighbor_bmm_ii = torch.bmm(data_neighbor_ii.transpose(1, 2), data_neighbor_ii)  # 2048 x 3 x 3
        data_average_ii = torch.unsqueeze(torch.mean(data_neighbor_ii, 1), 1)  # 2048 x 1 x 3
        data_average_bmm_ii = torch.bmm(data_average_ii.transpose(1, 2), data_average_ii)
        covariance_mat_ii = data_neighbor_bmm_ii / knn - data_average_bmm_ii
        # weight_normalize_unsqueeze = torch.stack(weight_normalize_unsqueeze)
        # print(weight_normalize.shape)
        covariance_mat_flat_ii = torch.unsqueeze(
            torch.cat((covariance_mat_ii[:, 0, :], covariance_mat_ii[:, 1, :], covariance_mat_ii[:, 2, :]), 1), 0
        )
        weight_normalize_mtx_ii = (torch.unsqueeze(weight_normalize, 2)).expand(-1, -1, 3)
        data_weight_average_ii = torch.unsqueeze(torch.sum(weight_normalize_mtx_ii * data_neighbor_ii, 1), 1)
        data_neighbor_weight_bmm_ii = torch.bmm(
            data_neighbor_ii.transpose(1, 2), weight_normalize_mtx_ii * data_neighbor_ii
        )
        data_weight_average_bmm_ii = torch.bmm(data_weight_average_ii.transpose(1, 2), data_weight_average_ii)
        weight_covariance_mat_ii = data_neighbor_weight_bmm_ii - data_weight_average_bmm_ii
        weight_covariance_mat_flat_ii = torch.unsqueeze(
            torch.cat(
                (
                    weight_covariance_mat_ii[:, 0, :],
                    weight_covariance_mat_ii[:, 1, :],
                    weight_covariance_mat_ii[:, 2, :],
                ),
                1,
            ),
            0,
        )

        covariance_mat.append(covariance_mat_flat_ii)
        # print(maxk_index.shape)
        weight_covariance_mat.append(weight_covariance_mat_flat_ii)
    covariance_mat_tensor = torch.cat(covariance_mat, dim=0)
    # print("weight_element_tensor shape: " , weight_element_tensor.shape)
    weight_covariance_mat_tensor = torch.cat(weight_covariance_mat, dim=0)
    covariance_mat_numpy = covariance_mat_tensor.data.cpu().numpy()
    weight_covariance_mat_numpy = weight_covariance_mat_tensor.data.cpu().numpy()
    print("covariance_mat_numpy shape: ", covariance_mat_numpy.shape)
    np.save(
        "MN40_test_cov_knn_" + str(knn) + ".npy",
        {b"data": seg_data_data, b"label": seg_data_label, b"weight_element": covariance_mat_numpy},
    )
    np.save(
        "MN40_test_weight_cov_knn_" + str(knn) + ".npy",
        {b"data": seg_data_data, b"label": seg_data_label, b"weight_element": weight_covariance_mat_numpy},
    )
    print("Done!")
    knn_index += 1
