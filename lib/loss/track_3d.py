import torch.nn as nn
import torch.nn.functional as F
import sys
import math

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class Track_3D_loss(nn.Module):

    def __init__(self, conf, verbose=True):

        super(Track_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.bbox_axis_head_lambda = 0 if not ('bbox_axis_head_lambda' in conf) else conf.bbox_axis_head_lambda
        self.bbox_3d_iou_lambda = 0 if not ('bbox_3d_iou_lambda' in conf) else conf.bbox_3d_iou_lambda
        self.bbox_3d_iou_focal = 1 if not ('bbox_3d_iou_focal' in conf) else conf.bbox_3d_iou_focal

        self.closeness_lambda = 0 if not ('closeness_lambda' in conf) else conf.closeness_lambda
        self.has_un = 0 if not ('has_un' in conf) else conf.has_un
        self.bbox_un_lambda = 0 if not ('bbox_un_lambda' in conf) else conf.bbox_un_lambda
        self.all_un = 0 if not ('all_un' in conf) else conf.all_un
        self.ind_un = 0 if not ('ind_un' in conf) else conf.ind_un

        self.decomp_alpha = False if not ('decomp_alpha' in conf) else conf.decomp_alpha
        self.decomp_trig_rot = False if not ('decomp_trig_rot' in conf) else conf.decomp_trig_rot

        self.track_scale_z3d = False if not ('track_scale_z3d' in conf) else conf.track_scale_z3d

        self.bbox_un_dynamic = False if not ('bbox_un_dynamic' in conf) else conf.bbox_un_dynamic

        self.motion_stats = None if not ('motion_stats' in conf) else conf.motion_stats
        self.infer_2d_from_3d = False if not ('infer_2d_from_3d' in conf) else conf.infer_2d_from_3d

        self.track_lambda = 0 if not ('track_lambda' in conf) else conf.track_lambda

        self.n_frames = 0

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.torch_bool = hasattr(torch, 'bool')
        self.torch_bool_type = torch.cuda.ByteTensor if not self.torch_bool else torch.cuda.BoolTensor
        self.verbose = verbose

    def forward(self, tracks_batch, imobjs, feat_size, rois=None, rois_3d=None, rois_3d_cen=None, post=None, key='gts', mask=None):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        error_x = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_y = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_z = torch.tensor(0).type(torch.cuda.FloatTensor)

        error_w = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_h = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_l = torch.tensor(0).type(torch.cuda.FloatTensor)

        error_r = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_he = torch.tensor(0).type(torch.cuda.FloatTensor)
        error_v = torch.tensor(0).type(torch.cuda.FloatTensor)

        track_loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        batch_size = len(tracks_batch)

        total_boxes = 0

        for bind in range(0, batch_size):

            tracks = tracks_batch[bind]

            if tracks is None:
                continue

            bbox_2d = tracks.box2ds[:, 0:4].detach().cpu().numpy()
            bbox_3d = tracks.Xs

            imobj = imobjs[bind]
            gts = imobj[key]

            #p2 = torch.from_numpy(imobj.p2).type(torch.cuda.FloatTensor)
            #p2_inv = torch.from_numpy(imobj.p2_inv).type(torch.cuda.FloatTensor)

            #p2_a = imobj.p2[0, 0].item()
            #p2_b = imobj.p2[0, 2].item()
            #p2_c = imobj.p2[0, 3].item()
            #p2_d = imobj.p2[1, 1].item()
            #p2_e = imobj.p2[1, 2].item()
            #p2_f = imobj.p2[1, 3].item()
            #p2_h = imobj.p2[2, 3].item()

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any() or bbox_2d.shape[0] == 0:
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            #head_mask = bbox_3d[:, 7] >= 0.5
            #bbox_3d = bbox_3d.clone()
            #bbox_3d.data[head_mask, 6] = snap_to_pi(bbox_3d.data[head_mask, 6] + math.pi)
            #bbox_3d.data[:, 6] = snap_to_pi(bbox_3d.data[:, 6])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                gts_val /= imobj.scale_factor

                ols = iou(bbox_2d, gts_val)
                ols_max = np.amax(ols, axis=1)
                targets = np.argmax(ols, axis=1)

                fg_inds = np.flatnonzero(ols_max >= self.best_thresh)

                if len(fg_inds):

                    target_2d = gts_val[targets[fg_inds], :]
                    target_3d = torch.from_numpy(gts_3d[targets[fg_inds], :]).type(torch.cuda.FloatTensor)

                    # reminder that format is
                    # [x, y, z, w, h, l, theta, head, vel]

                    ry3d_tar = target_3d[:, 10].clone()
                    ry3d_tar[target_3d[:, 15].type(self.torch_bool_type)] -= math.pi
                    ry3d_tar = snap_to_pi(ry3d_tar)

                    loss_w3d = F.smooth_l1_loss(bbox_3d[fg_inds, 3], target_3d[:, 3], reduction='none')
                    loss_h3d = F.smooth_l1_loss(bbox_3d[fg_inds, 4], target_3d[:, 4], reduction='none')
                    loss_l3d = F.smooth_l1_loss(bbox_3d[fg_inds, 5], target_3d[:, 5], reduction='none')

                    loss_x3d = F.smooth_l1_loss(bbox_3d[fg_inds, 0], target_3d[:, 7], reduction='none')
                    loss_y3d = F.smooth_l1_loss(bbox_3d[fg_inds, 1], target_3d[:, 8], reduction='none')
                    loss_z3d = F.smooth_l1_loss(bbox_3d[fg_inds, 2], target_3d[:, 9], reduction='none')

                    loss_rot = F.smooth_l1_loss(bbox_3d[fg_inds, 6], ry3d_tar, reduction='none')
                    loss_head = F.binary_cross_entropy(bbox_3d[fg_inds, 7], target_3d[:, 15], reduction='none')
                    loss_vel = F.smooth_l1_loss(bbox_3d[fg_inds, 8].clamp(min=0), target_3d[:, 16], reduction='none')

                    temp_loss = loss_w3d + loss_h3d + loss_l3d + loss_x3d + loss_y3d + loss_z3d + loss_rot \
                                  + loss_head + loss_vel

                    if self.track_scale_z3d != False:
                        temp_loss = temp_loss / (target_3d[:, 9].clamp(min=1)**(self.track_scale_z3d))

                    track_loss += temp_loss.sum()

                    error_x += (bbox_3d[fg_inds, 0] - target_3d[:, 7]).abs().sum()
                    error_y += (bbox_3d[fg_inds, 1] - target_3d[:, 8]).abs().sum()
                    error_z += (bbox_3d[fg_inds, 2] - target_3d[:, 9]).abs().sum()

                    error_w += (bbox_3d[fg_inds, 3] - target_3d[:, 3]).abs().sum()
                    error_h += (bbox_3d[fg_inds, 4] - target_3d[:, 4]).abs().sum()
                    error_l += (bbox_3d[fg_inds, 5] - target_3d[:, 5]).abs().sum()

                    error_r += (bbox_3d[fg_inds, 6] - ry3d_tar).abs().sum()
                    error_he += ((bbox_3d[fg_inds, 7] >= 0.5) != target_3d[:, 15].type(self.torch_bool_type)).sum()
                    error_v += (bbox_3d[fg_inds, 8].clamp(min=0) - target_3d[:, 16]).abs().sum()

                    total_boxes += len(fg_inds)

                    a = 1

                a = 1

        if total_boxes > 0:
            track_loss = self.track_lambda * track_loss / total_boxes

            error_x /= total_boxes
            error_y /= total_boxes
            error_z /= total_boxes

            error_w /= total_boxes
            error_h /= total_boxes
            error_l /= total_boxes

            error_r /= total_boxes
            error_he /= total_boxes
            error_v /= total_boxes

            if torch.isfinite(track_loss):
                loss += track_loss
                stats.append({'name': 'track', 'val': track_loss.item(), 'format': '{:0.4f}', 'group': 'loss'})

                stats.append({'name': 'tr_x', 'val': error_x.item(), 'format': '{:0.2f}', 'group': 'misc'})
                #stats.append({'name': 'tr_y', 'val': error_y.item(), 'format': '{:0.2f}', 'group': 'misc'})
                stats.append({'name': 'tr_z', 'val': error_z.item(), 'format': '{:0.2f}', 'group': 'misc'})

                #stats.append({'name': 'w_tr', 'val': error_w.item(), 'format': '{:0.2f}', 'group': 'misc'})
                #stats.append({'name': 'h_tr', 'val': error_h.item(), 'format': '{:0.2f}', 'group': 'misc'})
                stats.append({'name': 'tr_l', 'val': error_l.item(), 'format': '{:0.2f}', 'group': 'misc'})

                stats.append({'name': 'tr_r', 'val': error_r.item(), 'format': '{:0.2f}', 'group': 'misc'})
                #stats.append({'name': 'he_tr', 'val': error_he.item(), 'format': '{:0.2f}', 'group': 'misc'})
                stats.append({'name': 'tr_v', 'val': error_v.item(), 'format': '{:0.2f}', 'group': 'misc'})

        return loss, stats
