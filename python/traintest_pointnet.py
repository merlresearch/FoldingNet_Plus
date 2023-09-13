# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
traintest_pointnet in deepgeom

"""

import argparse
import os
import sys

import glog as logger
import torch
import torch.autograd
import torch.utils.data
from deepgeom.datasets import PointCloudDataset
from deepgeom.pointnet import (
    BoostedPointNetVanilla,
    BoostedPointPairNet,
    BoostedPointPairNet2,
    BoostedPointPairNetSuccessivePool,
    PointNetAttentionPool,
    PointNetBilinearPool,
    PointNetTplMatch,
    PointNetVanilla,
    PointPairNet,
)
from deepgeom.traintester import TrainTester
from deepgeom.utils import count_parameter_num

parser = argparse.ArgumentParser(sys.argv[0], description="PointNet Classification")

parser.add_argument("-0", "--train-pkl", type=str, help="path of the training pkl file")
parser.add_argument("-1", "--test-pkl", type=str, help="path of the testing pkl file")
parser.add_argument("-e", "--epoch", type=int, default=100, help="training epochs")
parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
parser.add_argument("--test-batch-size", type=int, default=32, help="testing batch size")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.5, help="Solver momentum")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
parser.add_argument(
    "--shuffle-point-order", type=str, default="no", help="whether/how to shuffle point order (no/offline/online)"
)
parser.add_argument("--log-dir", type=str, default="logs/tmp", help="log folder to save training stats as numpy files")
parser.add_argument(
    "--verbose_per_n_batch",
    type=int,
    default=10,
    help="log training stats to console every n batch (<=0 disables training log)",
)

args = parser.parse_args(sys.argv[1:])
args.script_folder = os.path.dirname(os.path.abspath(__file__))

args.cuda = torch.cuda.is_available()

print(str(args))
sys.stdout.flush()

#### Main
kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    PointCloudDataset(args.train_pkl, shuffle_point_order=args.shuffle_point_order),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    PointCloudDataset(args.test_pkl, shuffle_point_order="no"), batch_size=args.test_batch_size, shuffle=True, **kwargs
)

test_loader.__iter__()

net = PointNetVanilla(MLP_dims=(3, 64, 64, 64, 128, 1024), FC_dims=(1024, 512, 256, 40))
# net = PointNetConcatX(
#     MLP_dims=[3,64,64,64,128,1024],
#     concatFlags=(0,0,1,0,1),
#     FC_dims=(1024,512,256,40)
# )
# net = PointNetTplMatch(
#     MLP_dims=(3,64,64,64,128,1024),
#     C_tpls=40, M_points=50
# )
# net = PointNetAttentionPool(
#     MLP_dims=(3,64,64,64,128,256),
#     Attention_dims=(3,64,64,128,4),
#     FC_dims=(1024,512,256,40)
# ) #test_acc=83.8%
# net = PointNetBilinearPool(
#     MLP1_dims=(3,64,64,64,128,512),
#     FC1_dims=(512,256,128),
#     MLP2_dims=(3,64,64,64,128,256),
#     FC2_dims=(256,128,8),
#     FC_dims=(1024,512,256,40)
# ) #test_acc=86.4% #test_acc=86.7% if use SSR
# net = BoostedPointPairNet(
#     boost_factor=16,
#     dims=(6,64,64,128),
#     FC_dims=(128,64,40)
# )
# net = BoostedPointPairNet2(
#     boost_factor=16,
#     dims=(6,64,64,192),
#     FC_dims=(192,96,40),
#     # boost_pool_max=False,
#     # sym_pool_max=False
# )
# net = BoostedPointPairNetSuccessivePool(
#     boost_factor=16,
#     dims=(6,64,64,192),
#     FC_dims=(192,96,40),
#     sym_pool_max=False
# )
# net = BoostedPointNetVanilla(
#     boost_factor=8,
#     dims=(3,6,6,12,12,40),
#     FC_dims=(40,40,40,40),
#     boost_pool_max=False
# )

logger.info("Number of parameters={}".format(count_parameter_num(net.parameters())))
solver = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()
runner = TrainTester(
    net=net,
    solver=solver,
    total_epochs=args.epoch,
    cuda=args.cuda,
    log_dir=args.log_dir,
    verbose_per_n_batch=args.verbose_per_n_batch,
)
runner.run(train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn)
logger.info("Done!")
