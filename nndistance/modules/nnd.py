# Copyright (C) 2018-2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from functions.nnd import NNDFunction
from torch.nn.modules.module import Module


class NNDModule(Module):
    def forward(self, input1, input2):
        return NNDFunction()(input1, input2)
