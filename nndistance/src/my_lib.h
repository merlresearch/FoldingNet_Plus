// Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
// SPDX-License-Identifier: AGPL-3.0-or-later

void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx);

int nnd_forward(THFloatTensor *xyz1, THFloatTensor *xyz2, THFloatTensor *dist1, THFloatTensor *dist2, THIntTensor *idx1, THIntTensor *idx2);

int nnd_backward(THFloatTensor *xyz1, THFloatTensor *xyz2, THFloatTensor *gradxyz1, THFloatTensor *gradxyz2, THFloatTensor *graddist1, THFloatTensor *graddist2, THIntTensor *idx1, THIntTensor *idx2);
