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


class RPN_3D_loss(nn.Module):

    def __init__(self, conf, verbose=True):

        super(RPN_3D_loss, self).__init__()

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

        self.bbox_un_dynamic = False if not ('bbox_un_dynamic' in conf) else conf.bbox_un_dynamic

        self.motion_stats = None if not ('motion_stats' in conf) else conf.motion_stats
        self.infer_2d_from_3d = False if not ('infer_2d_from_3d' in conf) else conf.infer_2d_from_3d

        self.n_frames = 0

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.torch_bool = hasattr(torch, 'bool')
        self.torch_bool_type = torch.cuda.ByteTensor if not self.torch_bool else torch.cuda.BoolTensor
        self.verbose = verbose

    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, rois=None, rois_3d=None, rois_3d_cen=None, post=None, key='gts', mask=None):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        if mask is not None:
            mask = mask.detach().cpu().numpy().astype(bool)

        has_vel = False

        batch_size = cls.shape[0]

        prob_detach = prob.cpu().detach().numpy()

        if not self.infer_2d_from_3d:
            bbox_x = bbox_2d[:, :, 0]
            bbox_y = bbox_2d[:, :, 1]
            bbox_w = bbox_2d[:, :, 2]
            bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = torch.zeros(bbox_l3d.shape).type(torch.cuda.FloatTensor)

        if self.decomp_alpha:
            bbox_rsin = bbox_3d[:, :, 6]
            bbox_rcos = bbox_3d[:, :, 7]
            bbox_axis = bbox_3d[:, :, 8]
            bbox_head = bbox_3d[:, :, 9]

            if bbox_3d.shape[2] == 11 and not self.has_un:
                has_vel = True
                bbox_vel = bbox_3d[:, :, 10]
            if bbox_3d.shape[2] == 20 and not self.has_un:
                has_vel = True
                bbox_vel = bbox_3d[:, :, 19]

        if self.has_un and (self.bbox_un_lambda > 0 or self.bbox_3d_iou_lambda > 0):
            if self.decomp_alpha:
                bbox_un = bbox_3d[:, :, 11:].clamp(min=0.0005)
            else:
                bbox_un = bbox_3d[:, :, 7:].clamp(min=0.0005)

        else:
            bbox_un = torch.ones(bbox_z3d.shape, requires_grad=False)

        bbox_x3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_y3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_z3d_proj = torch.zeros(bbox_x3d.shape)

        labels = np.zeros(cls.shape[0:2])
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        if self.decomp_alpha:
            bbox_axis_tar = np.zeros(cls.shape[0:2])
            bbox_head_tar = np.zeros(cls.shape[0:2])
            bbox_rsin_tar = np.zeros(cls.shape[0:2])
            bbox_rcos_tar = np.zeros(cls.shape[0:2])

        if has_vel:
            bbox_vel_tar_dn = torch.zeros(cls.shape[0:2]).cpu()
            bbox_vel_tar = torch.zeros(cls.shape[0:2]).cpu()
            bbox_vel_act = torch.zeros(cls.shape[0:2]).cpu()

        bbox_x3d_proj_tar = torch.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = torch.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = torch.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        ious_2d = torch.zeros(cls.shape[0:2])
        ious_3d = torch.zeros(cls.shape[0:2])
        coords_abs_z = torch.zeros(cls.shape[0:2])
        coords_abs_ry = torch.zeros(cls.shape[0:2])

        # get all rois
        if rois is None:
            rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
            rois = rois.type(torch.cuda.FloatTensor)

        if rois.shape[0] > batch_size: rois = rois[:batch_size]
        if len(rois.shape) == 2: rois = rois.unsqueeze(0)

        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        if self.decomp_alpha:
            bbox_rsin_dn = bbox_rsin * self.bbox_stds[:, 11][0] + self.bbox_means[:, 11][0]
            bbox_rcos_dn = bbox_rcos * self.bbox_stds[:, 12][0] + self.bbox_means[:, 12][0]

        if has_vel:
            bbox_vel_dn = bbox_vel * self.bbox_stds[:, 13][0] + self.bbox_means[:, 13][0]

        if rois_3d is None:
            rois_3d = self.anchors[rois[:, 4].type(torch.cuda.LongTensor), :]
            rois_3d = torch.tensor(rois_3d, requires_grad=False).type(torch.cuda.FloatTensor)

        if rois_3d.shape[0] > batch_size: rois_3d = rois_3d[:batch_size]
        if len(rois_3d.shape) == 2: rois_3d = rois_3d.unsqueeze(0)

        # compute 3d transform
        widths = rois[:, :, 2] - rois[:, :, 0] + 1.0
        heights = rois[:, :, 3] - rois[:, :, 1] + 1.0
        ctr_x = rois[:, :, 0] + 0.5 * widths
        ctr_y = rois[:, :, 1] + 0.5 * heights

        if rois_3d_cen is None:
            bbox_x3d_dn = bbox_x3d_dn * widths + ctr_x
            bbox_y3d_dn = bbox_y3d_dn * heights + ctr_y
        else:

            if rois_3d_cen.shape[0] > batch_size: rois_3d_cen = rois_3d_cen[:batch_size]
            if len(rois_3d_cen.shape) == 2: rois_3d_cen = rois_3d_cen.unsqueeze(0)

            bbox_x3d_dn = bbox_x3d_dn * widths + rois_3d_cen[:, :, 0]
            bbox_y3d_dn = bbox_y3d_dn * heights + rois_3d_cen[:, :, 1]
        bbox_z3d_dn = rois_3d[:, :, 4] + bbox_z3d_dn
        bbox_w3d_dn = torch.exp(bbox_w3d_dn) * rois_3d[:, :, 5]
        bbox_h3d_dn = torch.exp(bbox_h3d_dn) * rois_3d[:, :, 6]
        bbox_l3d_dn = torch.exp(bbox_l3d_dn) * rois_3d[:, :, 7]

        if self.decomp_trig_rot:
            bbox_ry3d_dn = rois_3d[:, :, 8] + torch.asin(bbox_ry3d_dn.clamp(min=-1, max=1))
        else:
            bbox_ry3d_dn = rois_3d[:, :, 8] + bbox_ry3d_dn

        if self.decomp_alpha:
            bbox_rsin_dn = rois_3d[:, :, 9] + torch.asin(bbox_rsin_dn.clamp(min=-0.999, max=0.999))
            bbox_rcos_dn = rois_3d[:, :, 10] + torch.acos(bbox_rcos_dn.clamp(min=-0.999, max=0.999)) - math.pi/2

        if has_vel:
            bbox_vel_dn = rois_3d[:, :, 11] + bbox_vel_dn
            bbox_vel_dn = bbox_vel_dn.clamp(min=0)

        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            gts = imobj[key]

            p2 = torch.from_numpy(imobj.p2).type(torch.cuda.FloatTensor)
            p2_inv = torch.from_numpy(imobj.p2_inv).type(torch.cuda.FloatTensor)

            p2_a = imobj.p2[0, 0].item()
            p2_b = imobj.p2[0, 2].item()
            p2_c = imobj.p2[0, 3].item()
            p2_d = imobj.p2[1, 1].item()
            p2_e = imobj.p2[1, 2].item()
            p2_f = imobj.p2[1, 3].item()
            p2_h = imobj.p2[2, 3].item()

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                rois = rois.cpu()

                # bbox regression
                transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois[bind].numpy(), self.fg_thresh,
                                                  self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                  self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                                  tracker=rois[bind, :, 4].numpy(), rois_3d=rois_3d[bind].detach().cpu().numpy(),
                                                          rois_3d_cen=rois_3d_cen[bind].detach().cpu().numpy(), decomp_trig_rot=self.decomp_trig_rot)

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                if self.decomp_alpha:
                    # normalize 3d
                    transforms[:, 5:14] -= self.bbox_means[:, 4:13]
                    transforms[:, 5:14] /= self.bbox_stds[:, 4:13]
                else:
                    # normalize 3d
                    transforms[:, 5:12] -= self.bbox_means[:, 4:11]
                    transforms[:, 5:12] /= self.bbox_stds[:, 4:11]

                if mask is not None:
                    # ignore EVERYTHING not found in the previous frame
                    transforms[(~mask[bind]), 4] = 0

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)

                transforms = torch.from_numpy(transforms)
                raw_gt = torch.from_numpy(raw_gt)

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                if self.decomp_alpha:
                    bbox_axis_tar[bind, :] = raw_gt[:, 19]
                    bbox_head_tar[bind, :] = raw_gt[:, 20]
                    bbox_rsin_tar[bind, :] = transforms[:, 12]
                    bbox_rcos_tar[bind, :] = transforms[:, 13]

                if has_vel and raw_gt.shape[1] == 22:
                    vel_tars = raw_gt[:, 21]
                    vel_valid = vel_tars >= 0
                    bbox_vel_tar_dn[bind, vel_valid] = raw_gt[vel_valid, 21]
                    bbox_vel_tar[bind, vel_valid] = (transforms[vel_valid, 14] - self.bbox_means[0, 13])/self.bbox_stds[0, 13]
                    bbox_vel_act[bind, vel_valid] = 1

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois[bind].shape[0]*self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois[bind].shape[0]*self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:


                    if not self.infer_2d_from_3d:

                        # compile deltas pred
                        deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis ], bbox_y[bind, :, np.newaxis],
                                               bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

                        # compile deltas targets
                        deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                        bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                       axis=1)

                        # move to gpu
                        deltas_2d_tar = torch.tensor(deltas_2d_tar, requires_grad=False).type(torch.cuda.FloatTensor)

                        means = self.bbox_means[0, :]
                        stds = self.bbox_stds[0, :]

                        rois = rois.cuda()

                        coords_2d = bbox_transform_inv(rois[bind], deltas_2d, means=means, stds=stds)
                        coords_2d_tar = bbox_transform_inv(rois[bind], deltas_2d_tar, means=means, stds=stds)

                        ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]

                    rois_3d_fg = rois_3d[bind, fg_inds, :]
                    if len(rois_3d_fg.shape) == 1: rois_3d_fg = rois_3d_fg.unsqueeze(0)

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
                    bbox_z3d_dn_fg = bbox_z3d_dn[bind, fg_inds]
                    bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
                    bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
                    bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
                    bbox_ry3d_dn_fg = bbox_ry3d_dn[bind, fg_inds]

                    if self.decomp_alpha:
                        #axis_cos_mask = bbox_axis_tar[bind, fg_inds] == 0
                        axis_sin_mask = bbox_axis_tar[bind, fg_inds] == 1
                        head_pos_mask = bbox_head_tar[bind, fg_inds] == 1

                        if not self.torch_bool:
                            axis_sin_mask = torch.from_numpy(np.array(axis_sin_mask, dtype=np.uint8))
                            head_pos_mask = torch.from_numpy(np.array(head_pos_mask, dtype=np.uint8))

                        bbox_ry3d_dn_fg = bbox_rcos_dn[bind, fg_inds]
                        bbox_ry3d_dn_fg[axis_sin_mask] = bbox_rsin_dn[bind, fg_inds][axis_sin_mask]
                        bbox_ry3d_dn_fg[head_pos_mask] = bbox_ry3d_dn_fg[head_pos_mask] + math.pi

                        #bbox_ry3d_dn_tar = raw_gt[fg_inds, 18]
                        #bbox_ry3d_dn_tar[axis_sin_mask] = raw_gt[fg_inds, 17][axis_sin_mask]

                    # re-scale all 2D back to original
                    bbox_x3d_dn_fg /= imobj['scale_factor']
                    bbox_y3d_dn_fg /= imobj['scale_factor']

                    #coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_y3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_z3d_dn_fg[np.newaxis,:]), dim=0)
                    #coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)

                    #coords_3d = torch.mm(p2_inv, coords_2d)

                    z3d = bbox_z3d_dn_fg - p2_h
                    x3d = ((z3d + p2_h) * bbox_x3d_dn_fg - p2_b * (z3d) - p2_c) / p2_a
                    y3d = ((z3d + p2_h) * bbox_y3d_dn_fg - p2_e * (z3d) - p2_f) / p2_d

                    bbox_x3d_proj[bind, fg_inds] = x3d
                    bbox_y3d_proj[bind, fg_inds] = y3d
                    bbox_z3d_proj[bind, fg_inds] = z3d

                    if self.bbox_3d_iou_lambda > 0:

                        gt_x3d = raw_gt[fg_inds, 5 + 7].type(torch.cuda.FloatTensor)
                        gt_y3d = raw_gt[fg_inds, 5 + 8].type(torch.cuda.FloatTensor)
                        gt_z3d = raw_gt[fg_inds, 5 + 9].type(torch.cuda.FloatTensor)
                        gt_w3d = raw_gt[fg_inds, 8].type(torch.cuda.FloatTensor)
                        gt_h3d = raw_gt[fg_inds, 9].type(torch.cuda.FloatTensor)
                        gt_l3d = raw_gt[fg_inds, 10].type(torch.cuda.FloatTensor)
                        gt_rotY = raw_gt[fg_inds, 5 + 10].type(torch.cuda.FloatTensor)

                        verts_gts, corners_3d_gt = project_3d(p2, gt_x3d, gt_y3d, gt_z3d, gt_w3d, gt_h3d, gt_l3d, gt_rotY, return_3d=True)
                        bbox_ry3d_dn_fg_after = convertAlpha2Rot(bbox_ry3d_dn_fg, z3d, x3d)
                        verts, corners_3d = project_3d(p2, x3d, y3d, z3d, bbox_w3d_dn_fg, bbox_h3d_dn_fg, bbox_l3d_dn_fg, bbox_ry3d_dn_fg_after, return_3d=True)

                        corners_3d = corners_3d.detach().cpu().numpy()
                        corners_3d_gt = corners_3d_gt.detach().cpu().numpy()

                        vol = (gt_h3d * gt_l3d * gt_w3d + bbox_h3d_dn_fg * bbox_l3d_dn_fg * bbox_w3d_dn_fg).detach().cpu().numpy()

                        # compute 3D IoU
                        for i, fgind in enumerate(fg_inds):
                            iou_bev_cur, iou_3d_cur = iou3d(corners_3d[i], corners_3d_gt[i], vol[i])
                            ious_3d[bind, fgind] = iou_3d_cur

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                    bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_z3d_dn_tar = rois_3d_fg[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                    bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_ry3d_dn_tar = rois_3d_fg[:, 8] + bbox_ry3d_dn_tar

                    if self.decomp_alpha:

                        bbox_ry3d_dn_tar = raw_gt[fg_inds, 11]
                        #bbox_ry3d_dn_tar[axis_sin_mask] = raw_gt[fg_inds, 17][axis_sin_mask]

                        bbox_ry3d_dn_tar = bbox_ry3d_dn_tar.cuda()

                    coords_abs_z[bind, fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
                    coords_abs_ry[bind, fg_inds] = torch.abs(bbox_ry3d_dn_tar - snap_to_pi(bbox_ry3d_dn_fg))

            else:

                bg_inds = np.arange(0, rois[bind].shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)


                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC


            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = cls.argmax(dim=2).cpu().detach().numpy()

        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            if self.verbose: stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels = torch.tensor(labels, requires_grad=False)
        labels = labels.view(-1).type(torch.cuda.LongTensor)

        labels_weight = torch.tensor(labels_weight, requires_grad=False)
        labels_weight = labels_weight.view(-1).type(torch.cuda.FloatTensor)

        cls = cls.view(-1, cls.shape[2])

        if self.cls_2d_lambda and mask is None:

            # cls loss
            active = labels_weight > 0

            if np.any(active.cpu().numpy()):

                loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                stats.append({'name': 'cls', 'val': loss_cls.detach(), 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            bbox_weights = torch.tensor(bbox_weights, requires_grad=False).type(torch.cuda.FloatTensor).view(-1)

            active = bbox_weights > 0

            if self.ind_un:
                bbox_un = bbox_un[:, :].view(-1, 3)
            else:
                bbox_un = bbox_un[:, :].view(-1)

            if self.bbox_3d_iou_lambda > 0 and mask is None:

                ious_3d = ious_3d.view(-1)

                # bbox_un_bg = ious_3d[active] < 0.5
                # bbox_un_fg = ious_3d[active] >= 0.5
                # loss_bbox_un_bg = torch.abs(torch.log((1 - ious_3d[active][bbox_un_bg])) - torch.log((1 - bbox_un[active][bbox_un_bg])))
                # loss_bbox_un_fg = torch.abs(torch.log((ious_3d[active][bbox_un_fg])) - torch.log((bbox_un[active][bbox_un_fg])))
                # loss_bbox_un = (loss_bbox_un_bg.sum() + loss_bbox_un_fg.sum()) / fg_inds.shape[0]
                loss_bbox_un = F.smooth_l1_loss(bbox_un[active], ious_3d[active], reduction='none').mean()
                #loss_bbox_un = torch.abs(bbox_un[active] - ious_3d[active]).mean() * self.bbox_3d_iou_lambda

                loss += loss_bbox_un

                bbox_weights = bbox_weights * ((ious_3d) ** self.bbox_3d_iou_focal)

                stats.append({'name': 'i3d', 'val': loss_bbox_un.detach(), 'format': '{:0.4f}', 'group': 'loss'})
                stats.append({'name': 'iou3d', 'val': ious_3d[active & torch.isfinite(ious_3d)].detach().mean(), 'format': '{:0.2f}', 'group': 'acc'})
                stats.append({'name': 'i3d', 'val': bbox_un[active].mean().item(), 'format': '{:0.2f}', 'group': 'misc'})

            if has_vel:
                bbox_vel_act = bbox_vel_act.type(self.torch_bool_type).view(-1)
                bbox_vel_tar = bbox_vel_tar.cuda().view(-1)
                bbox_vel_tar_dn = bbox_vel_tar_dn.cuda().view(-1)

                active_vel = bbox_vel_act * active

                if active_vel.any():
                    bbox_vel = bbox_vel.view(-1)
                    bbox_vel_dn = bbox_vel_dn.view(-1)

                    loss_vel = F.smooth_l1_loss(bbox_vel[active_vel], bbox_vel_tar[active_vel], reduction='none').mean()

                    acc = (bbox_vel_dn[active_vel] - bbox_vel_tar_dn[active_vel]).abs().mean()

                    loss += loss_vel
                    flname = 'fl' if post is None else 'fl' + post
                    stats.append({'name': 'fl', 'val': loss_vel.item(), 'format': '{:0.4f}', 'group': 'loss'})
                    stats.append({'name': flname, 'val': acc.item(), 'format': '{:0.4f}', 'group': 'misc'})
                    a = 1

            if self.bbox_2d_lambda and mask is None:

                # bbox loss 2d
                bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                if self.verbose: stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            # bbox center distance
            bbox_x3d_proj_tar = bbox_x3d_proj_tar.view(-1)
            bbox_y3d_proj_tar = bbox_y3d_proj_tar.view(-1)
            bbox_z3d_proj_tar = bbox_z3d_proj_tar.view(-1)

            bbox_x3d_proj = bbox_x3d_proj[:, :].view(-1)
            bbox_y3d_proj = bbox_y3d_proj[:, :].view(-1)
            bbox_z3d_proj = bbox_z3d_proj[:, :].view(-1)

            cen_dist = torch.sqrt(((bbox_x3d_proj[active] - bbox_x3d_proj_tar[active])**2)
                                  + ((bbox_y3d_proj[active] - bbox_y3d_proj_tar[active])**2)
                                  + ((bbox_z3d_proj[active] - bbox_z3d_proj_tar[active])**2))

            cen_match = (cen_dist <= 0.20).type(torch.cuda.FloatTensor)

            if self.verbose: stats.append({'name': 'cen', 'val': cen_dist.mean().detach(), 'format': '{:0.2f}', 'group': 'misc'})
            stats.append({'name': 'match', 'val': cen_match.mean().detach(), 'format': '{:0.3f}', 'group': 'misc'})

            closeness = torch.abs(bbox_z3d_proj_tar[active]).clamp(min=5, max=100) ** self.closeness_lambda

            if self.bbox_3d_lambda:

                # bbox loss 3d
                bbox_x3d_tar = torch.tensor(bbox_x3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y3d_tar = torch.tensor(bbox_y3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_z3d_tar = torch.tensor(bbox_z3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_w3d_tar = torch.tensor(bbox_w3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_h3d_tar = torch.tensor(bbox_h3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_l3d_tar = torch.tensor(bbox_l3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_ry3d = bbox_ry3d[:, :].view(-1)

                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none')

                if self.decomp_alpha:

                    bbox_rsin_tar = torch.tensor(bbox_rsin_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                    bbox_rcos_tar = torch.tensor(bbox_rcos_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                    bbox_axis_tar = torch.tensor(bbox_axis_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                    bbox_head_tar = torch.tensor(bbox_head_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                    bbox_axis_sin_mask = bbox_axis_tar[active] == 1
                    bbox_head_pos_mask = bbox_head_tar[active] == 1

                    bbox_rsin = bbox_rsin[:, :].view(-1)
                    bbox_rcos = bbox_rcos[:, :].view(-1)
                    bbox_axis = bbox_axis[:, :].view(-1)
                    bbox_head = bbox_head[:, :].view(-1)

                    loss_bbox_rsin = F.smooth_l1_loss(bbox_rsin[active], bbox_rsin_tar[active], reduction='none')
                    loss_bbox_rcos = F.smooth_l1_loss(bbox_rcos[active], bbox_rcos_tar[active], reduction='none')
                    loss_axis = F.binary_cross_entropy(bbox_axis[active], bbox_axis_tar[active], reduction='none')
                    loss_head = F.binary_cross_entropy(bbox_head[active], bbox_head_tar[active], reduction='none')

                    loss_bbox_ry3d = loss_bbox_rcos
                    loss_bbox_ry3d[bbox_axis_sin_mask] = loss_bbox_rsin[bbox_axis_sin_mask]

                    # compute axis accuracy
                    fg_points = bbox_axis[active].detach().cpu().numpy()
                    fg_labels = bbox_axis_tar[active].detach().cpu().numpy()
                    if self.verbose: stats.append({'name': 'axis', 'val': ((fg_points >= 0.5) == fg_labels).mean(), 'format': '{:0.2f}', 'group': 'acc'})

                    # compute axis accuracy
                    fg_points = bbox_head[active].detach().cpu().numpy()
                    fg_labels = bbox_head_tar[active].detach().cpu().numpy()
                    if self.verbose: stats.append({'name': 'head', 'val': ((fg_points >= 0.5) == fg_labels).mean(), 'format': '{:0.2f}', 'group': 'acc'})

                loss_bbox_x3d = (loss_bbox_x3d * bbox_weights[active])
                loss_bbox_y3d = (loss_bbox_y3d * bbox_weights[active])
                loss_bbox_z3d = (loss_bbox_z3d * bbox_weights[active])
                loss_bbox_w3d = (loss_bbox_w3d * bbox_weights[active])
                loss_bbox_h3d = (loss_bbox_h3d * bbox_weights[active])
                loss_bbox_l3d = (loss_bbox_l3d * bbox_weights[active])
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active])

                if self.decomp_alpha:
                    loss_axis = (loss_axis * bbox_weights[active])
                    loss_head = (loss_head * bbox_weights[active])

                if self.bbox_un_dynamic:

                    loss_bbox_3d_init = float((loss_bbox_w3d[torch.isfinite(loss_bbox_w3d)].mean()
                                        + loss_bbox_h3d[torch.isfinite(loss_bbox_h3d)].mean()
                                        + loss_bbox_l3d[torch.isfinite(loss_bbox_l3d)].mean()
                                        + loss_bbox_ry3d[torch.isfinite(loss_bbox_ry3d)].mean()
                                        + loss_bbox_x3d[torch.isfinite(loss_bbox_x3d)].mean()
                                        + loss_bbox_y3d[torch.isfinite(loss_bbox_y3d)].mean()
                                        + loss_bbox_z3d[torch.isfinite(loss_bbox_z3d)].mean()).item())*self.bbox_3d_lambda

                    if self.decomp_alpha:
                        loss_bbox_3d_init += float((loss_axis[torch.isfinite(loss_axis)].mean()
                                                    + loss_head[torch.isfinite(loss_head)].mean()).item())*self.bbox_axis_head_lambda

                    if self.n_frames == 0:
                        self.bbox_un_lambda = loss_bbox_3d_init
                        self.n_frames += 1
                    else:
                        self.n_frames = min(100, self.n_frames + 1)
                        self.bbox_un_lambda = loss_bbox_3d_init/self.n_frames + self.bbox_un_lambda*(self.n_frames - 1)/self.n_frames
                        #self.bbox_un_lambda = loss_bbox_3d_init*(1 - 0.90) + self.bbox_un_lambda*0.90

                if self.all_un and self.ind_un and self.bbox_un_lambda > 0:
                    loss_bbox_x3d = (loss_bbox_x3d)
                    loss_bbox_y3d = (loss_bbox_y3d)
                    loss_bbox_z3d = (loss_bbox_z3d) * bbox_un[active, 0]
                    loss_bbox_w3d = (loss_bbox_w3d)
                    loss_bbox_h3d = (loss_bbox_h3d)
                    loss_bbox_l3d = (loss_bbox_l3d)
                    loss_bbox_ry3d = (loss_bbox_ry3d) * bbox_un[active, 1]
                elif self.all_un and self.bbox_un_lambda > 0:
                    loss_bbox_x3d = (loss_bbox_x3d) * bbox_un[active]
                    loss_bbox_y3d = (loss_bbox_y3d) * bbox_un[active]
                    loss_bbox_z3d = (loss_bbox_z3d) * bbox_un[active]
                    loss_bbox_w3d = (loss_bbox_w3d) * bbox_un[active]
                    loss_bbox_h3d = (loss_bbox_h3d) * bbox_un[active]
                    loss_bbox_l3d = (loss_bbox_l3d) * bbox_un[active]
                    loss_bbox_ry3d = (loss_bbox_ry3d) * bbox_un[active]
                    if self.decomp_alpha:
                        loss_axis = (loss_axis) * bbox_un[active]
                        loss_head = (loss_head) * bbox_un[active]

                if mask is None:
                    bbox_3d_loss = (loss_bbox_w3d[torch.isfinite(loss_bbox_w3d)].mean()
                                    + loss_bbox_h3d[torch.isfinite(loss_bbox_h3d)].mean()
                                    + loss_bbox_l3d[torch.isfinite(loss_bbox_l3d)].mean()
                                    + loss_bbox_ry3d[torch.isfinite(loss_bbox_ry3d)].mean())

                    if self.decomp_alpha:
                        bbox_3d_loss += (loss_axis.mean() + loss_head.mean())*self.bbox_axis_head_lambda
                else:
                    stats = []
                    loss = torch.tensor(0).type(torch.cuda.FloatTensor)
                    bbox_3d_loss = torch.tensor(0).type(torch.cuda.FloatTensor)

                if self.bbox_3d_proj_lambda == 0:
                    bbox_3d_loss += (loss_bbox_x3d[torch.isfinite(loss_bbox_x3d)].mean()
                                     + loss_bbox_y3d[torch.isfinite(loss_bbox_y3d)].mean()
                                     + loss_bbox_z3d[torch.isfinite(loss_bbox_z3d)].mean())

                bbox_3d_loss *= self.bbox_3d_lambda

                # simple gradient clipping
                #bbox_3d_loss = bbox_3d_loss.clamp(min=0, max=10)

                loss += bbox_3d_loss

                if mask is not None:
                    coords_abs_z = coords_abs_z.view(-1)
                    stats.append({'name': 'fl_bbox_3d', 'val': bbox_3d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})
                    stats.append({'name': 'fl_z', 'val': coords_abs_z[active].detach().mean(), 'format': '{:0.2f}', 'group': 'misc'})
                else:
                    stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_un_lambda > 0 and mask is None:

                if self.ind_un:
                    loss_bbox_un = (1 - bbox_un[active, 2]).mean()
                else:
                    loss_bbox_un = (1 - bbox_un[active]).mean()

                    #ious_3d = ious_3d.view(-1)

                    #bbox_un_bg = ious_3d[active] < 0.5
                    #bbox_un_fg = ious_3d[active] >= 0.5

                    #loss_bbox_un_bg = torch.abs(torch.log((1 - ious_3d[active][bbox_un_bg])) - torch.log((1 - bbox_un[active][bbox_un_bg])))
                    #loss_bbox_un_fg = torch.abs(torch.log((ious_3d[active][bbox_un_fg])) - torch.log((bbox_un[active][bbox_un_fg])))

                    #loss_bbox_un = (loss_bbox_un_bg.sum() + loss_bbox_un_fg.sum()) / fg_inds.shape[0]
                    #loss_bbox_un = F.smooth_l1_loss(bbox_un[active], ious_3d[active], reduction='none').mean()
                    #loss_bbox_un = torch.abs(bbox_un[active] - ious_3d[active]).mean()

                    a = 1

                if not self.all_un:

                    loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none') / closeness
                    loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none') / closeness
                    loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none') / closeness
                    loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none') / closeness
                    loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none') / closeness
                    loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none') / closeness
                    loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none') / closeness

                    loss_bbox_x3d = (loss_bbox_x3d) * bbox_un[active]
                    loss_bbox_y3d = (loss_bbox_y3d) * bbox_un[active]
                    loss_bbox_z3d = (loss_bbox_z3d) * bbox_un[active]
                    loss_bbox_w3d = (loss_bbox_w3d) * bbox_un[active]
                    loss_bbox_h3d = (loss_bbox_h3d) * bbox_un[active]
                    loss_bbox_l3d = (loss_bbox_l3d) * bbox_un[active]
                    loss_bbox_ry3d = (loss_bbox_ry3d) * bbox_un[active]

                    loss_bbox_un += (loss_bbox_w3d[torch.isfinite(loss_bbox_w3d)].mean()
                                    + loss_bbox_h3d[torch.isfinite(loss_bbox_h3d)].mean()
                                    + loss_bbox_l3d[torch.isfinite(loss_bbox_l3d)].mean()
                                    + loss_bbox_ry3d[torch.isfinite(loss_bbox_ry3d)].mean())

                    loss_bbox_un += (loss_bbox_x3d[torch.isfinite(loss_bbox_x3d)].mean()
                                         + loss_bbox_y3d[torch.isfinite(loss_bbox_y3d)].mean()
                                         + loss_bbox_z3d[torch.isfinite(loss_bbox_z3d)].mean())

                loss_bbox_un *= self.bbox_un_lambda

                loss += loss_bbox_un
                stats.append({'name': 'un', 'val': loss_bbox_un.detach(), 'format': '{:0.4f}', 'group': 'loss'})

                if self.ind_un:
                    stats.append({'name': 'conf', 'val': bbox_un[active, 2].mean().detach(), 'format': '{:0.2f}', 'group': 'misc'})
                else:
                    stats.append({'name': 'conf', 'val': bbox_un[active].mean().detach(), 'format': '{:0.2f}', 'group': 'misc'})


            if self.bbox_3d_proj_lambda and mask is None:

                #cen_loss = torch.sqrt(((bbox_x3d_proj[active] - bbox_x3d_proj_tar[active]) ** 2)
                #                      + ((bbox_y3d_proj[active] - bbox_y3d_proj_tar[active]) ** 2)
                #                      + ((bbox_z3d_proj[active] - bbox_z3d_proj_tar[active]) ** 2))

                #cen_loss = cen_loss / closeness
                #cen_loss = cen_loss[torch.isfinite(cen_loss)].mean()

                #bbox_3d_proj_loss = cen_loss * self.bbox_3d_proj_lambda

                loss_bbox_x3d_proj = F.smooth_l1_loss(bbox_x3d_proj[active], bbox_x3d_proj_tar[active], reduction='none') / closeness
                loss_bbox_y3d_proj = F.smooth_l1_loss(bbox_y3d_proj[active], bbox_y3d_proj_tar[active], reduction='none') / closeness
                loss_bbox_z3d_proj = F.smooth_l1_loss(bbox_z3d_proj[active], bbox_z3d_proj_tar[active], reduction='none') / closeness

                bbox_3d_proj_loss = (loss_bbox_x3d_proj[torch.isfinite(loss_bbox_x3d_proj)].mean()
                                    + loss_bbox_y3d_proj[torch.isfinite(loss_bbox_y3d_proj)].mean()
                                    + loss_bbox_z3d_proj[torch.isfinite(loss_bbox_z3d_proj)].mean())

                # simple gradient clipping
                #bbox_3d_proj_loss = bbox_3d_proj_loss.clamp(min=0.0, max=10)

                loss += bbox_3d_proj_loss
                stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            if mask is None:
                coords_abs_z = coords_abs_z.view(-1)
                zname = 'z' if post is None else 'z' + post
                stats.append({'name': zname, 'val': coords_abs_z[active].detach().mean(), 'format': '{:0.2f}', 'group': 'misc'})

                coords_abs_ry = coords_abs_ry.view(-1)
                rname = 'ry' if post is None else 'ry' + post
                stats.append({'name': rname, 'val': coords_abs_ry[active].detach().mean(), 'format': '{:0.2f}', 'group': 'misc'})

                if not self.infer_2d_from_3d:
                    ious_2d = ious_2d.view(-1)
                    stats.append({'name': 'iou', 'val': ious_2d[active & torch.isfinite(ious_2d)].detach().mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda and mask is None and (ious_2d[active] != 0).any() and not self.infer_2d_from_3d:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                iou_2d_loss = iou_2d_loss[torch.isfinite(iou_2d_loss)].mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                if self.verbose: stats.append({'name': 'iou', 'val': iou_2d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})


        return loss, stats
