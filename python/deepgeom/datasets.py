# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PointCloudDataset with covariance matrix in deepgeom
"""

import argparse
import os
import sys

import glog as logger
import numpy as np
from torch.utils.data import Dataset


class PointCloudAugTrainDataset(Dataset):
    def __init__(self, pkl_path, shuffle_point_order="no", train_or_test=0):
        self.shuffle_point_order = shuffle_point_order

        logger.info("loading: " + pkl_path)

        raw_data = np.load(pkl_path, encoding="bytes").item()
        self.all_data = raw_data[b"data"]  # [BxNx3]
        if shuffle_point_order == "preprocess":
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        self.weight_element1 = raw_data[b"weight_element1"]
        self.weight_element2 = raw_data[b"weight_element2"]
        self.weight_element3 = raw_data[b"weight_element3"]

        logger.info(
            "pkl loaded: data "
            + str(self.all_data.shape)
            + ", weight_element1 "
            + str(self.weight_element1.shape)
            + ", weight element2 "
            + str(self.weight_element2.shape)
        )
        logger.check_eq(len(self.all_data.shape), 3, "data field should of size BxNx3!")
        logger.check_eq(self.all_data.shape[-1], 3, "data field the last dimension size should be 3!")
        logger.check_eq(self.weight_element1.shape[-1], 3, "weight element1 filed the last dimension size should be 3!")
        logger.check_eq(self.weight_element2.shape[-1], 3, "weight element2 filed the last dimension size should be 3!")
        logger.check_eq(self.weight_element3.shape[-1], 3, "weight element3 filed the last dimension size should be 3!")
        logger.check_eq(
            self.all_data.shape[0],
            self.weight_element1.shape[0],
            "data field and weight element field should have the same size along the first dimension!",
        )

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        if self.shuffle_point_order == "online":
            np.random.shuffle(self.all_data[idx])
        return {
            "data": self.all_data[idx],
            "augment1": self.weight_element1[idx],
            "augment2": self.weight_element2[idx],
            "augment3": self.weight_element3[idx],
        }


class PointCloudAugTestDataset(Dataset):
    def __init__(self, pkl_path, shuffle_point_order="no", train_or_test=0):
        self.shuffle_point_order = shuffle_point_order

        logger.info("loading: " + pkl_path)
        raw_data = np.load(pkl_path, encoding="bytes").item()
        self.all_data = raw_data[b"data"]  # [BxNx3]
        if shuffle_point_order == "preprocess":
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        self.all_label = np.asarray(raw_data[b"label"], dtype=np.int64)
        self.all_weight = raw_data[b"weight_element"]

        logger.info(
            "pkl loaded: data "
            + str(self.all_data.shape)
            + ", label "
            + str(self.all_label.shape)
            + ", weight element "
            + str(self.all_weight.shape)
        )
        logger.check_eq(len(self.all_data.shape), 3, "data field should of size BxNx3!")
        logger.check_eq(self.all_data.shape[-1], 3, "data field the last dimension size should be 3!")
        logger.check_eq(
            self.all_weight.shape[-1], 9, "weight element field for test the last dimension size should be 9!"
        )
        logger.check_eq(len(self.all_label.shape), 1, "label field should be one dimensional!")
        logger.check_eq(
            self.all_data.shape[0],
            self.all_label.shape[0],
            "data field and label field should have the same size along the first dimension!",
        )

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        if self.shuffle_point_order == "online":
            np.random.shuffle(self.all_data[idx])
        return {"data": self.all_data[idx], "label": self.all_label[idx], "weight": self.all_weight[idx]}


def main(args):
    import utils

    modelnet = PointCloudDataset(pkl_path="../../data/modelNet_test.pkl", shuffle_point_order="online")
    for i, ith in enumerate(modelnet):
        print("{:d}: {:d}".format(i, ith["label"]))

        X = ith["data"]
        rdir = np.random.rand(
            3,
        )
        rdir /= np.linalg.norm(rdir.reshape(-1))
        clr = (X.dot(rdir)).reshape(-1)
        clr = (clr - clr.min()) / (clr.max() - clr.min())
        clr = np.round(clr * 20)
        fig = utils.vis_pts(X, clr=clr, cmap="tab20")
        utils.plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
