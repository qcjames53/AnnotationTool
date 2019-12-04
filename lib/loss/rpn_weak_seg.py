import torch.nn as nn
import torch.nn.functional as F
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class RPN_Weak_Seg_loss(nn.Module):

    def __init__(self, conf):

        super(RPN_Weak_Seg_loss, self).__init__()

        self.num_classes = len(conf['lbls']) + 1
        self.feat_stride = conf['feat_stride']
        self.weak_seg_lambda = conf['weak_seg_lambda']
        self.fg_fraction = conf['fg_fraction']

        self.lbls = conf['lbls']
        self.ilbls = conf['ilbls']

        self.min_gt_vis = conf['min_gt_vis']
        self.min_gt_h = conf['min_gt_h']
        self.max_gt_h = conf['max_gt_h']


    def forward(self, seg, imobjs):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        IGN_FLAG = 3000

        batch_size = seg.shape[0]

        # compute feature resolution
        feat_size = [seg.shape[2], seg.shape[3]]

        labels_weak = np.zeros([batch_size, 1, feat_size[0], feat_size[1]], dtype=float)
        labels_weight_weak = np.ones([batch_size, 1, feat_size[0], feat_size[1]], dtype=float)

        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            gts = imobj['gts']

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt['bbox_full'] for gt in gts]))

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]

            # accumulate labels
            box_lbls = np.array([gt['cls'] for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            multicls_region = np.zeros(feat_size, dtype=bool)

            for gt_ign in gts_ign:

                x1 = int(np.clip(np.floor(gt_ign[0] / self.feat_stride), a_min=0, a_max=feat_size[1] - 1))
                y1 = int(np.clip(np.floor(gt_ign[1] / self.feat_stride), a_min=0, a_max=feat_size[0] - 1))
                x2 = int(np.clip(np.ceil(gt_ign[2] / self.feat_stride), a_min=0, a_max=feat_size[1] - 1))
                y2 = int(np.clip(np.ceil(gt_ign[3] / self.feat_stride), a_min=0, a_max=feat_size[0] - 1))

                labels_weak[bind, 0, y1:y2 + 1, x1:x2 + 1] = IGN_FLAG
                labels_weight_weak[bind, 0, y1:y2 + 1, x1:x2 + 1] = 0

            for gtind, gt_val in enumerate(gts_val):

                x1 = int(np.clip(np.floor(gt_val[0] / self.feat_stride), a_min=0, a_max=feat_size[1] - 1))
                y1 = int(np.clip(np.floor(gt_val[1] / self.feat_stride), a_min=0, a_max=feat_size[0] - 1))
                x2 = int(np.clip(np.ceil(gt_val[2] / self.feat_stride), a_min=0, a_max=feat_size[1] - 1))
                y2 = int(np.clip(np.ceil(gt_val[3] / self.feat_stride), a_min=0, a_max=feat_size[0] - 1))

                lbl = box_lbls[gtind]

                # any other classes already labeled there? then we will need to ignore it later
                # as this is technically ambiguous region with more information
                multicls_region |= ((labels_weak[bind, 0, :, :] != IGN_FLAG) & (labels_weak[bind, 0, :, :] > 0)
                                    & (labels_weak[bind, 0, :, :] != lbl))

                labels_weak[bind, 0, y1:y2 + 1, x1:x2 + 1] = lbl

            labels_weak[bind, 0, multicls_region] = IGN_FLAG
            labels_weight_weak[bind, 0, multicls_region] = 0

        # convert to tensor
        labels_weak = torch.tensor(labels_weak, requires_grad=False).type(torch.cuda.LongTensor)
        labels_weight_weak = torch.tensor(labels_weight_weak, requires_grad=False).type(torch.cuda.FloatTensor)

        # flatten everything
        seg = flatten_tensor(seg).view(-1, self.num_classes)
        labels_weak = flatten_tensor(labels_weak).view(-1)
        labels_weight_weak = flatten_tensor(labels_weight_weak).view(-1)

        # re-weight
        fg_bools = (labels_weak > 0) & (labels_weak != IGN_FLAG)
        bg_bools = (labels_weak ==0)

        fg_num = fg_bools.sum().item()
        bg_num = bg_bools.sum().item()

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight_weak[fg_bools] = fg_weight
                labels_weight_weak[bg_bools] = 1.0

            else:
                labels_weight_weak[bg_bools] = 1.0

        active = labels_weight_weak > 0

        if np.any(active) and self.weak_seg_lambda:

            loss_seg = F.cross_entropy(seg[active, :], labels_weak[active], reduction='none', ignore_index=IGN_FLAG)
            loss_seg = (loss_seg * labels_weight_weak[active])

            # simple gradient clipping
            loss_seg = loss_seg.clamp(min=0, max=2000)

            # take mean and scale lambda
            loss_seg = loss_seg.mean()
            loss_seg *= self.weak_seg_lambda

            loss += loss_seg

            stats.append({'name': 'weak_seg', 'val': loss_seg, 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
