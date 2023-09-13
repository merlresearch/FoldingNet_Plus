#!/usr/bin/env bash

# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# ../python/traintest_pointnet.py
PY_NAME=$1
# ../logs/
LOG_DIR=$2
JOB_NAME=${LOG_DIR##*/}
mkdir -p $LOG_DIR
TRAIN_SET=/data37/tian/datasets/ModelNet/prepared/modelNet40_train_file_16nn_GM.npy
TEST_SET=/data37/tian/datasets/ModelNet/prepared/modelNet40_test_file_16nn_GM.npy
# srun -p clusterNew -X -D $PWD --job-name=$JOB_NAME --gres gpu:1 python $PY_NAME -0 $TRAIN_SET -1 $TEST_SET --log-dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/all.log
python $PY_NAME -0 $TRAIN_SET -1 $TEST_SET --log-dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/all.log
