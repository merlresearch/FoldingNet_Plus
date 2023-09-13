# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
utils.py in deepgeom

"""

import argparse
import errno
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, proj3d


def check_exist_or_mkdirs(path):
    """thread-safe mkdirs if not exist"""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def vis_pts(pts, clr, cmap):
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)

    ax.set_alpha(0)
    ax.set_aspect("equal")
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim, max_lim)
    ax.set_ylim3d(min_lim, max_lim)
    ax.set_zlim3d(min_lim, max_lim)

    if clr is None:
        M = ax.get_proj()
        _, _, clr = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], M)
        clr = (clr - clr.min()) / (clr.max() - clr.min())

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=clr, zdir="x", s=20, cmap=cmap, edgecolors="k")
    return fig


def count_parameter_num(params):
    cnt = 0
    for p in params:
        cnt += np.prod(p.size())
    return cnt


class TrainTestMonitor(object):
    def __init__(self, log_dir, plot_loss_max=4.0, plot_extra=False):
        assert os.path.exists(log_dir)

        stats_test = np.load(os.path.join(log_dir, "stats_test.npz"))
        stats_train_running = np.load(os.path.join(log_dir, "stats_train_running.npz"))

        self.title = os.path.basename(log_dir)
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        plt.title(self.title)

        # Training loss
        iter_loss = stats_train_running["iter_loss"]
        self.ax1.plot(iter_loss[:, 0], iter_loss[:, 1], "-", label="train loss", color="r", linewidth=2)
        self.ax1.set_ylim([0, plot_loss_max])
        self.ax1.set_xlabel("iteration")
        self.ax1.set_ylabel("loss")

        # Test accuracy
        iter_acc = stats_test["iter_acc"]
        max_accu_pos = np.argmax(iter_acc[:, 1])
        test_label = "max test accuracy {:.3f} @ {}".format(iter_acc[max_accu_pos, 1], max_accu_pos + 1)
        self.ax2.plot(iter_acc[:, 0], iter_acc[:, 1], "o--", label=test_label, color="b", linewidth=2)
        self.ax2.set_ylabel("accuracy")

        if plot_extra:
            # Training accuracy
            iter_acc = stats_train_running["iter_acc"]
            self.ax2.plot(iter_acc[:, 0], iter_acc[:, 1], "--", label="train accuracy", color="b", linewidth=0.8)
            # Test loss
            iter_loss = stats_test["iter_loss"]
            self.ax1.plot(iter_loss[:, 0], iter_loss[:, 1], "--", label="test loss", color="r", linewidth=0.8)

        self.ax1.legend(loc="upper left", framealpha=0.8)
        self.ax2.legend(loc="lower right", framealpha=0.8)
        self.fig.show()


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
