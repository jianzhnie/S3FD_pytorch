#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
from itertools import product as product
import math


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg.INPUT_SIZE
        self.variance = cfg.VARIANCE or [0.1]
        self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        #self.feature_maps = feature_maps

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size/ self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size

                mean += [cx, cy, s_k, s_k]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    import sys, os
    this_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(this_dir, '../../data'))
    from config import cfg
    p = PriorBox(cfg)
    out = p.forward()
    print(out.size())
    sum = 0
    feature_maps = cfg.FEATURE_MAPS
    for x in feature_maps:
        sum += x**2
    print(sum)

