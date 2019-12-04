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


class Eye_loss(nn.Module):

    def __init__(self, conf):
        super(Eye_loss, self).__init__()


    def forward(self, affine):

        stats = []

        gt_eye = np.tile(np.eye(affine.shape[1])[np.newaxis,:], [affine.shape[0],1,1])
        gt_eye = torch.from_numpy(gt_eye).type(torch.cuda.FloatTensor)

        loss = torch.abs(affine.flatten() - gt_eye.flatten()).mean()

        stats.append({'name': 'eye', 'val': loss, 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
