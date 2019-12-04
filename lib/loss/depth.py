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


class Depth_loss(nn.Module):

    def __init__(self, conf):
        super(Depth_loss, self).__init__()

        self.depth_lambda = conf.depth_lambda


    def forward(self, depth, prob, imobjs):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        batch_size = depth.shape[0]
        gt_maps = []

        # read depth maps in
        for bind, imobj in enumerate(imobjs):
            #gt_map = imread(imobj.depth_path)[:, :, 0].astype(np.float32)
            gt_map = np.array(Image.open(imobj.depth_path)).astype(np.float32)/256.0 - 1
            gt_map = cv2.resize(gt_map, (depth.shape[3], depth.shape[2]), interpolation=cv2.INTER_NEAREST)
            gt_maps.append(gt_map)


        # convert to tensor
        gt_maps = torch.from_numpy(np.array(gt_maps)).type(torch.cuda.FloatTensor).unsqueeze(1)

        # flatten both
        gt_maps = flatten_tensor(gt_maps)
        depth = flatten_tensor(depth)
        gt_maps = gt_maps.view(-1)
        depth = depth.view(-1) #, depth.shape[2])

        active = (gt_maps >= 0) & (gt_maps < 60)

        #loss_depth = F.cross_entropy(depth[active], gt_maps[active], reduction='none')
        loss_depth = F.smooth_l1_loss(depth[active], gt_maps[active], reduction='none')
        loss_depth = (loss_depth).mean()

        loss += loss_depth * self.depth_lambda
        stats.append({'name': 'depth', 'val': loss_depth, 'format': '{:0.4f}', 'group': 'loss'})

        depth_pred = depth #depth.argmax(dim=1)

        coords_abs_z = torch.abs(depth_pred[active] - gt_maps[active])
        coords_abs_z = np.mean(coords_abs_z.detach().cpu().numpy())
        stats.append({'name': 'z', 'val': coords_abs_z, 'format': '{:0.2f}', 'group': 'misc'})

        return loss, stats
