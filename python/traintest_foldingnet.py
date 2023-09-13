# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
traintest_foldingnet in deepgeom

"""

import argparse
import os
import sys

import glog as logger
import torch
import torch.autograd
import torch.utils.data
from deepgeom.datasets import PointCloudNewDataset  # load PointCloudIsomapDataset instead of PointCloudDataset
from deepgeom.foldingnet import (
    ChamfersDistance,
    ChamfersDistance2,
    ChamfersDistance3,
    FoldingNetDCT2D,
    FoldingNetDCTGraph,
    FoldingNetPlus,
    FoldingNetShrink,
    FoldingNetVanilla,
)
from deepgeom.traintester import TrainTester
from deepgeom.utils import count_parameter_num


def main(args):

    # Load data
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        PointCloudNewDataset(
            args.train_pkl, shuffle_point_order=args.shuffle_point_order, train_or_test=args.train_or_test
        ),  # load PointCloudIsomapDataset instead of PointCloudDataset
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        PointCloudNewDataset(
            args.test_pkl, shuffle_point_order="no", train_or_test=args.train_or_test
        ),  # load PointCloudIsomapDataset instead of PointCloudDataset
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    print("args.channel_number: {}".format(args.channel_number))
    if args.version == "plus":
        net = FoldingNetPlus(
            MLP_dims=(3, 64, 64, 64, 128, 1024),
            FC_dims=(1024, 512, 512),
            grid_dims=(45, 45),  # defaul 45
            Folding1_dims=(514, 512, 512, 3),  # for foldingnet
            Folding2_dims=(515, 512, 512, 3),  # for foldingnet
            Folding3_dims=(2028, 512, 512, 3),  # for foldingnet
            Weight1_dims=(2537, 512, 512, args.channel_number),  # for weight matrix estimation 45x45+512 = 2537
            Weight3_dims=(512 + args.channel_number, 1024, 1024, 2025),
            knn=args.knn,
            sigma=args.sigma,
        )
    elif args.version == "shrink":
        print("args.band_low: {}".format(args.band_low))
        print("args.band_high: {}".format(args.band_high))

        net = FoldingNetShrink(
            MLP_dims=(3, 64, 64, 64, 128, 1024),
            FC_dims=(1024, 512, 512),
            grid_dims=(45, 45),  # defaul 45
            Folding1_dims=(514, 512, 512, 3),  # for foldingnet
            Folding2_dims=(515, 512, 512, 3),  # for foldingnet
            Folding3_dims=(2028, 512, 512, 3),  # for foldingnet
            Weight1_dims=(
                512 + args.band_high - args.band_low,
                512,
                512,
                args.channel_number,
            ),  # for weight matrix estimation 45x45+512 = 2537
            Weight3_dims=(512 + args.channel_number, 1024, 1024, 2025),
            knn=args.knn,
            sigma=args.sigma,
            bandwidth=[args.band_low, args.band_high],
        )
    elif args.version == "dct":
        net = FoldingNetDCT2D(
            MLP_dims=(3, 64, 64, 64, 128, 1024),
            FC_dims=(1024, 512, 512),
            grid_dims=(45, 45),  # defaul 45
            Folding1_dims=(512 + 100, 512, 512, 3),  # for foldingnet
            Folding2_dims=(515, 512, 512, 3),  # for foldingnet
            Folding3_dims=(2028, 512, 512, 3),  # for foldingnet
            Weight1_dims=(512 + 100, 512, 512, args.channel_number),  # for weight matrix estimation 45x45+512 = 2537
            Weight3_dims=(512 + args.channel_number, 1024, 1024, 2025),
            knn=args.knn,
            sigma=args.sigma,
        )
    elif args.version == "dct_graph":
        net = FoldingNetDCTGraph(
            MLP_dims=(3, 64, 64, 64, 128, 1024),
            FC_dims=(1024, 512, 512),
            grid_dims=(45, 45),  # defaul 45
            Folding1_dims=(512 + 100, 512, 512, 3),  # for foldingnet
            Folding2_dims=(515, 512, 512, 3),  # for foldingnet
            Folding3_dims=(3 + 256, 512, 512, 3),  # for foldingnet
            Weight1_dims=(512 + 100, 512, 512, args.channel_number),  # for weight matrix estimation 45x45+512 = 2537
            Weight3_dims=(512 + args.channel_number, 1024, 1024, 256),
            knn=args.knn,
            sigma=args.sigma,
        )
    else:
        net = FoldingNetVanilla(
            MLP_dims=(3, 64, 64, 64, 128, 1024),
            FC_dims=(1024, 512, 512),
            grid_dims=(45, 45),  # defaul 45
            Folding1_dims=(514, 512, 512, 3),  # for foldingnet
            Folding2_dims=(515, 512, 512, 3),  # for foldingnet
            Weight1_dims=(2537, 512, 512, args.channel_number),  # for weight matrix estimation 45x45+512 = 2537
            Weight3_dims=(512 + args.channel_number, 1024, 1024, 2025),
            knn=args.knn,
            sigma=args.sigma,
        )
    #
    logger.info("Number of parameters={}".format(count_parameter_num(net.parameters())))
    # solver = torch.optim.SGD(net.parameters(),
    #                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    solver = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = ChamfersDistance()
    runner = TrainTester(
        net=net,
        solver=solver,
        total_epochs=args.epoch,
        cuda=args.cuda,
        log_dir=args.log_dir,
        verbose_per_n_batch=args.verbose_per_n_batch,
        haar_coeff=args.haar_coeff,
        start_epochs=args.start_epochs,
    )

    if args.restore_checkpoint_filename is not None:
        path = os.path.join(args.log_dir, args.restore_checkpoint_filename)
        net.load_state_dict(torch.load(path))
        print("restoring the checkpoint at: {}".format(path))

    runner.run(train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, train_or_test=args.train_or_test)

    if args.output_filename is not None:
        torch.save(net.state_dict(), os.path.join(args.log_dir, args.output_filename))

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0], description="FoldingNet Autoencoder")

    parser.add_argument(
        "-0",
        "--train-pkl",
        type=str,
        default="../dataset/old_train_with_modelnet.npy",
        help="path of the training pkl file",
    )
    parser.add_argument(
        "-1",
        "--test-pkl",
        type=str,
        default="../dataset/new_test_with_shapenet.npy",
        help="path of the testing pkl file",
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=300, help="training epochs"  # change the default setting
    )  # 278
    parser.add_argument(
        "-t", "--train_or_test", type=int, default=0, help="train-0-test-1-both-2"  # train or test the network
    )  # 278
    parser.add_argument("-k", "--knn", type=int, default=96, help="number of neighboring points")
    parser.add_argument(
        "-sigma", "--sigma", type=int, default=0.008, help="radius of kernels for calculating weight matrix"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1, help="testing batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")  # 1e-4 default
    parser.add_argument("--momentum", type=float, default=0.9, help="Solver momentum")  # 0.9 default
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="weight decay")  # 1e-6 default
    parser.add_argument(
        "--shuffle-point-order", type=str, default="no", help="whether/how to shuffle point order (no/offline/online)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/tmp", help="log folder to save training stats as numpy files"
    )
    parser.add_argument(
        "--verbose_per_n_batch",
        type=int,
        default=10,
        help="log training stats to console every n batch (<=0 disables training log)",
    )

    parser.add_argument("--start_epochs", type=int, default=0, help="start epochs")
    parser.add_argument("--restore_checkpoint_filename", type=str, default=None, help="param_weight_k96_08")
    parser.add_argument("--output-filename", type=str, default="param_weight_k96_08.pkl", help="param_weight_k96_08")
    parser.add_argument(
        "--channel_number",
        type=int,
        default=128,
        help="log training stats to console every n batch (<=0 disables training log)",
    )

    parser.add_argument("--version", type=str, default="vanilla", help="advanced version")
    parser.add_argument("--haar_coeff", type=float, default=0.5, help="advanced version")

    parser.add_argument("--band_low", type=int, default=0, help="bandwidth low")
    parser.add_argument("--band_high", type=int, default=100, help="bandwidth high")

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    args.cuda = torch.cuda.is_available()

    print(str(args))
    sys.stdout.flush()

    main(args)
