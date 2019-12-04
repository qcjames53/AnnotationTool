import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class Affine_loss(nn.Module):

    def __init__(self, conf):
        super(Affine_loss, self).__init__()


    def forward(self, affine, imobjs):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        batch_size = affine.shape[0]

        for bind in range(0, batch_size):

            imobj = imobjs[bind]

            if imobj.affine_gt is None: continue

            affine_gt = torch.from_numpy(imobj.affine_gt).type(torch.cuda.FloatTensor)

            loss += torch.abs(affine[bind, :].flatten() - affine_gt.flatten()).mean()

        stats.append({'name': 'affine', 'val': loss, 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
