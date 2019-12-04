import torch.nn as nn
import torch.nn.functional as F
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class RPN_3D_flow_loss(nn.Module):

    def __init__(self, conf):

        super(RPN_3D_flow_loss, self).__init__()

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
        self.moving_target = conf.moving_target

        self.crop_size = conf.crop_size
        self.scatter_scale = 1 if not ('scatter_scale' in conf) else conf.scatter_scale
        self.scatter_stride = 1 if not ('scatter_scale' in conf) else conf.scatter_scale

        self.video_det = 0 if not ('video_det' in conf) else conf.video_det
        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda
        self.iou_3d_lambda = conf.iou_3d_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.pose_means = conf.pose_means
        self.pose_stds = conf.pose_stds

        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds


    def forward(self, flow, imobjs, feat_size):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        flow_x3d = flow[0, :, 0]
        flow_y3d = flow[1, :, 0]
        flow_z3d = flow[2, :, 0]

        #flow_x3d = (flow_x3d) * self.bbox_stds[0, 4]
        #flow_y3d = (flow_y3d) * self.bbox_stds[0, 5]
        #flow_z3d = (flow_z3d) * self.bbox_stds[0, 6]

        flow_x3d_tar = np.zeros(flow_x3d.shape[0:2])
        flow_z3d_tar = np.zeros(flow_x3d.shape[0:2])
        flow_y3d_tar = np.zeros(flow_x3d.shape[0:2])

        active = np.zeros(flow_x3d.shape[0:2])

        p2 = imobjs[0].p2
        p2_inv = np.linalg.inv(p2)

        sf = imobjs[0].scale_factor

        # get all rois
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=False)

        gts0 = imobjs[0].gts_pre
        gts1 = imobjs[0].gts

        # filter gts
        igns0, rmvs0 = determine_ignores(gts0, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)
        igns1, rmvs1 = determine_ignores(gts1, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

        if len(gts0) <= 0: return loss, stats

        # accumulate boxes
        gts0_box = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts0]))
        gts0 = np.array([(gt.center_3d, gt.track) for gt in gts0])
        gts1 = np.array([(gt.center_3d, gt.track) for gt in gts1])

        if len(gts0) <= 0: return loss, stats

        # filter out irrelevant cls, and ignore cls
        gts0_box = gts0_box[(rmvs0 == False) & (igns0 == False), :]
        gts0 = gts0[(rmvs0 == False) & (igns0 == False), :]
        #gts1 = gts1[(rmvs1 == False) & (igns1 == False), :]

        pose_dx, pose_dy, pose_dz, pose_rx, pose_ry, pose_rz = compute_rel_pose(imobjs[0].pose_pre, imobjs[0].pose)

        '''
        for all rois in im0, match the src ground truth
        then search the ground truth by id in im1
        if match, then compute:
            flow_x = x1 - x0 - vo_x
            flow_y = y1 - y0 - vo_y
            flow_z = z1 - z0 - vo_z
        '''

        if gts0_box.shape[0] > 0:
            ols = iou(rois, gts0_box)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= self.best_thresh]
            #gt_best_ols = gt_best_ols[gt_best_ols >= self.best_thresh]

            fg_inds = np.flatnonzero(ols_max >= self.fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)

            fg_labels = np.zeros(ols.shape[0], dtype=bool)
            fg_labels[fg_inds] = True

            tracks = [gts0[t][1] for t in targets]

            for gtind0, gt0 in enumerate(gts0):

                for gtind1, gt1 in enumerate(gts1):

                    if gt0[1] == gt1[1]:

                        x0 = gt0[0][0]
                        y0 = gt0[0][1]
                        z0 = gt0[0][2]

                        x0 += pose_dx
                        y0 += pose_dy
                        z0 += pose_dz

                        rig_loc0 = p2.dot(np.array([x0, y0, z0, 1]))
                        rig_loc0[0:2] *= sf
                        rig_loc0[0:2] /= rig_loc0[2]

                        x1 = gt1[0][0]
                        y1 = gt1[0][1]
                        z1 = gt1[0][2]

                        loc1 = p2.dot(np.array([x1, y1, z1, 1]))
                        loc1[0:2] *= sf
                        loc1[0:2] /= loc1[2]

                        # x1 = x0 + pose + flow_x
                        # ==> flow_x = x1 - x0 - pose
                        flow_x = loc1[0] - rig_loc0[0]
                        flow_y = loc1[1] - rig_loc0[1]
                        flow_z = loc1[2] - rig_loc0[2]

                        flow_x3d_tar[(targets == gtind0) & fg_labels] = flow_x
                        flow_y3d_tar[(targets == gtind0) & fg_labels] = flow_y
                        flow_z3d_tar[(targets == gtind0) & fg_labels] = flow_z

                        active[(targets == gtind0) & fg_labels] = 1

        if np.any(active):

            flow_x3d_tar = torch.from_numpy(flow_x3d_tar).cuda().type(torch.cuda.FloatTensor)
            flow_y3d_tar = torch.from_numpy(flow_y3d_tar).cuda().type(torch.cuda.FloatTensor)
            flow_z3d_tar = torch.from_numpy(flow_z3d_tar).cuda().type(torch.cuda.FloatTensor)

            active = torch.from_numpy(active).cuda().type(torch.cuda.LongTensor)

            loss_flow_x = torch.abs(flow_x3d_tar[active] - flow_x3d[active]).mean()
            loss_flow_y = torch.abs(flow_y3d_tar[active] - flow_y3d[active]).mean()
            loss_flow_z = torch.abs(flow_z3d_tar[active] - flow_z3d[active]).mean()

            loss += (loss_flow_x + loss_flow_y + loss_flow_z)*1

            stats.append({'name': 'flow_x', 'val': loss_flow_x.detach().cpu().numpy(), 'format': '{:0.4f}', 'group': 'misc'})
            stats.append({'name': 'flow_y', 'val': loss_flow_y.detach().cpu().numpy(), 'format': '{:0.4f}', 'group': 'misc'})
            stats.append({'name': 'flow_z', 'val': loss_flow_z.detach().cpu().numpy(), 'format': '{:0.4f}', 'group': 'misc'})

            stats.append({'name': 'flow', 'val': loss.detach().cpu().numpy(), 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
