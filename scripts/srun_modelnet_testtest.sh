#!/usr/bin/env bash
# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

PY_NAME=$1
LOG_DIR=$2
JOB_NAME=${LOG_DIR##*/}
mkdir -p $LOG_DIR
TRAIN_SET=../data/modelNet_test.pkl
TEST_SET=../data/modelNet_test.pkl
srun -p clusterNew -X -D $PWD --job-name=$JOB_NAME --gres gpu:1 python $PY_NAME -0 $TRAIN_SET -1 $TEST_SET --log-dir $LOG_DIR 2>&1 | tee ${LOG_DIR}/all.log
