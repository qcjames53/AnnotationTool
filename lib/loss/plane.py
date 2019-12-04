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
from lib.math_3d import *


class Plane_loss(nn.Module):

    def __init__(self, conf):
        super(Plane_loss, self).__init__()

        self.gp_max = conf.gp_max

        self.plane_means = conf.plane_means
        self.plane_stds = conf.plane_stds

    def forward(self, plane, imobjs):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        batch_size = plane.shape[0]

        center_errors = np.empty(0)

        for bind in range(0, batch_size):

            imobj = imobjs[bind]

            if imobj.plane_gt is None: continue

            gts_gp = [gt for gt in imobj.gts if gt.cls in ['Pedestrian', 'Car', 'Cyclist'] and not gt.ign and gt.bbox_3d[2] > 0 and gt.bbox_3d[2] <= self.gp_max]

            gt = edict()
            gt.center_3d = [0, 1.65, 0]
            gt.bbox_3d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            center_2d = imobj.p2.dot(np.array(gt.center_3d + [1]))
            gt.bbox_3d[0] = center_2d[0]
            gt.bbox_3d[1] = center_2d[1]
            gt.bbox_3d[2] = center_2d[2]
            gts_gp.append(gt)

            for gtind in range(len(gts_gp)):
                gt = gts_gp[gtind]

                x3d = gt.center_3d[0]
                y3d = gt.center_3d[1]
                z3d = gt.center_3d[2]
                h3d = gt.bbox_3d[4]

                # get bottom center
                y3d_bot = y3d + h3d / 2

                # project bottom center
                x2d_bot, y2d_bot, z2d_bot = project_3d_point(imobj.p2, x3d, y3d_bot, z3d)

                # store corresponding
                gt.bottom_3d = np.array([x3d, y3d_bot, z3d])
                gt.bottom_2d = np.array([x2d_bot, y2d_bot, z2d_bot])

            # form train and test
            points3d = np.empty([0, 3])
            points2d = np.empty([0, 2])

            for gtind, gt in enumerate(gts_gp):
                points2d = np.vstack((points2d, gt.bottom_2d[np.newaxis, :2]))
                points3d = np.vstack((points3d, gt.bottom_3d[np.newaxis, :3]))

            # de-norm
            plane_dn = plane[bind, :].detach().cpu().numpy()
            plane_dn *= self.plane_stds[0, :].max()
            plane_dn += self.plane_means[0, :]

            # compute errors for gts
            center_errors = np.concatenate((center_errors, plane_eval(imobj.p2_inv, plane_dn, points2d, points3d)))

            plane_gt = imobj.plane_gt
            plane_gt -= self.plane_means[0, :]
            plane_gt /= self.plane_stds[0,:].max()
            plane_gt = torch.from_numpy(imobj.plane_gt).type(torch.cuda.FloatTensor)

            loss += torch.abs(plane[bind, :].flatten() - plane_gt.flatten()).mean()

        stats.append({'name': 'plane', 'val': loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})
        stats.append({'name': 'gp_cen', 'val': center_errors.mean(), 'format': '{:0.4f}', 'group': 'misc'})

        return loss, stats
