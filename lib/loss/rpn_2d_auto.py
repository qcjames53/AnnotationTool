import torch.nn as nn
import torch.nn.functional as F
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *
from lib.nms.gpu_nms import gpu_nms


class RPN_2D_auto_loss(nn.Module):

    def __init__(self, num_classes, anchors, bbox_means, bbox_stds, feat_stride, fg_fraction, box_samples, ign_thresh,
                 nms_thres, fg_thresh, bg_thresh_lo, bg_thresh_hi, best_thresh, hard_negatives, focal_loss,
                 natural_cls, moving_target, lbls, ilbls, min_gt_vis, min_gt_h, max_gt_h, iou_2d_loss, bbox_2d_loss,
                 gpu=0):

        super(RPN_2D_auto_loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = anchors.shape[0]
        self.anchors = anchors
        self.bbox_means = bbox_means
        self.bbox_stds = bbox_stds
        self.feat_stride = feat_stride
        self.fg_fraction = fg_fraction
        self.box_samples = box_samples
        self.ign_thresh = ign_thresh
        self.nms_thres = nms_thres
        self.fg_thresh = fg_thresh
        self.bg_thresh_lo = bg_thresh_lo
        self.bg_thresh_hi = bg_thresh_hi
        self.best_thresh = best_thresh
        self.hard_negatives = hard_negatives
        self.focal_loss = focal_loss
        self.natural_cls = natural_cls
        self.moving_target = moving_target

        self.iou_2d_loss = iou_2d_loss
        self.bbox_2d_loss = bbox_2d_loss

        self.lbls = lbls
        self.ilbls = ilbls

        self.min_gt_vis = min_gt_vis
        self.min_gt_h = min_gt_h
        self.max_gt_h = max_gt_h

        self.gpu = gpu

    def forward(self, clses, probs, bbox_2d, imobjs):

        stats = []

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        for phase in range(0, len(clses)):

            first_phase = phase == 0
            last_phase = phase == (len(clses) - 1)

            cls = clses[phase]
            prob = probs[phase]

            bg_lo = self.bg_thresh_lo[phase]
            bg_hi = self.bg_thresh_hi[phase]
            fg_lo = self.fg_thresh[phase]
            ign_lo = self.ign_thresh[phase]
            best_lo = self.best_thresh[phase]

            batch_size = cls.shape[0]

            # compute feature resolution
            feat_size = [int(bbox_2d[0].shape[2] / self.num_anchors), bbox_2d[0].shape[3]]

            # flatten everything
            cls = flatten_tensor(cls)
            prob = flatten_tensor(prob)

            prob_detach = prob.cpu().detach().numpy()

            labels = np.zeros(cls.shape[0:2])
            labels_weight = np.zeros(cls.shape[0:2])

            labels_scores = np.zeros(cls.shape[0:2])

            bbox_x = flatten_tensor(bbox_2d[0])
            bbox_y = flatten_tensor(bbox_2d[1])
            bbox_w = flatten_tensor(bbox_2d[2])
            bbox_h = flatten_tensor(bbox_2d[3])

            bbox_x_tar = np.zeros(cls.shape[0:2])
            bbox_y_tar = np.zeros(cls.shape[0:2])
            bbox_w_tar = np.zeros(cls.shape[0:2])
            bbox_h_tar = np.zeros(cls.shape[0:2])

            bbox_weights = np.zeros(cls.shape[0:2])

            coords_2d_iou = torch.zeros(cls.shape[0:2])

            for bind in range(0, batch_size):

                imobj = imobjs[bind]
                gts = imobj['gts']

                # get all rois
                rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)

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

                if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                    # bbox regression
                    transforms, ols = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), fg_lo, ign_lo, bg_lo,
                                                      bg_hi, best_lo, anchors=self.anchors)

                    # normalize 2d
                    transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                    transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                    labels_fg = transforms[:, 4] > 0
                    labels_bg = transforms[:, 4] < 0
                    labels_ign = transforms[:, 4] == 0

                    fg_inds = np.flatnonzero(labels_fg)
                    bg_inds = np.flatnonzero(labels_bg)
                    ign_inds = np.flatnonzero(labels_ign)

                    transforms = torch.from_numpy(transforms).cuda()

                    labels[bind, fg_inds] = transforms[fg_inds, 4]
                    labels[bind, ign_inds] = IGN_FLAG
                    labels[bind, bg_inds] = 0

                    bbox_x_tar[bind, :] = transforms[:, 0]
                    bbox_y_tar[bind, :] = transforms[:, 1]
                    bbox_w_tar[bind, :] = transforms[:, 2]
                    bbox_h_tar[bind, :] = transforms[:, 3]

                    # ----------------------------------------
                    # box sampling
                    # ----------------------------------------

                    if self.box_samples == np.inf:
                        fg_num = len(fg_inds)
                        bg_num = len(bg_inds)

                    else:
                        fg_num = min(round(self.box_samples * self.fg_fraction), len(fg_inds))
                        bg_num = min(round(self.box_samples - fg_num), len(bg_inds))

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

                    # compute 2D IoU
                    if fg_num > 0:

                        # compile deltas pred
                        deltas_2d = torch.cat((bbox_x[bind, :, :], bbox_y[bind, :, :],
                                               bbox_w[bind, :, :], bbox_h[bind, :, :]), dim=1)

                        # compile deltas targets
                        deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                        bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                       axis=1)

                        # move to gpu
                        deltas_2d_tar = torch.tensor(deltas_2d_tar).type(torch.cuda.FloatTensor)
                        rois = rois.type(torch.cuda.FloatTensor)

                        means = self.bbox_means[0, :]
                        stds = self.bbox_stds[0, :]

                        coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)
                        coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)

                        coords_2d_iou[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')

                        if self.moving_target or last_phase:

                            # recompute the transforms using
                            coords_2d_detach = coords_2d.cpu().detach().numpy()

                            # determine the best anchors
                            gt_best_rois = np.argmax(ols, axis=0)
                            gt_best_ols = np.amax(ols, axis=0)

                            # filter too low best threshold
                            found = gt_best_ols >= best_lo
                            best_lbls = box_lbls[found]
                            best_inds = gt_best_rois[found]
                            best_ols = gt_best_ols[found]

                            # original fg
                            prev_fg = (labels[bind, :] > 0) & (labels[bind, :] != IGN_FLAG)

                            if best_inds.shape[0] > 0:

                                # do a light version of NMS using transformed boxes
                                ols_best = iou(coords_2d_detach[best_inds, :], coords_2d_detach)

                                # deterime the suppressed (can be ignored) and potential false positives indices
                                ign_inds = np.flatnonzero((np.argmax(ols_best, axis=0) >= self.nms_thres) & prev_fg)
                                fp_inds = np.flatnonzero((np.argmax(ols_best, axis=0) < self.nms_thres) & prev_fg)

                                # ignore any suppressed foreground
                                labels[bind, ign_inds] = IGN_FLAG
                                labels_weight[bind, ign_inds] = 0

                                # mark false positives as background
                                labels[bind, fp_inds] = 0
                                labels_weight[bind, fp_inds] = BG_ENC

                                # mark all best as foreground
                                labels[bind, best_inds] = best_lbls
                                labels_weight[bind, best_inds] = FG_ENC


                else:

                    bg_inds = np.arange(0, rois.shape[0])

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

            if len(fg_inds_all) > 0:
                acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
                stats.append({'name': 'fg_p' + str(phase), 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

            if len(bg_inds_all) > 0:
                acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
                stats.append({'name': 'bg_p' + str(phase), 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

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
                    labels_weight[bg_inds_unravel] = bg_weights

                # re-weight fg
                if fg_num > 0:
                    fg_scores = labels_scores[fg_inds_unravel]
                    fg_weights = (1 - fg_scores) ** self.focal_loss
                    weights_sum += np.sum(fg_weights)
                    labels_weight[fg_inds_unravel] = fg_weights

                # adjust the mean
                if box_samples > 0:
                    mean_weight = weights_sum/box_samples
                    labels_weight /= mean_weight

                else:
                    raise ValueError('no samples for loss!')

            elif self.natural_cls:
                labels_weight[active_inds_unravel] = 1.0

            else:

                if fg_num > 0:

                    fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                    labels_weight[fg_inds_unravel] = fg_weight
                    labels_weight[bg_inds_unravel] = 1.0

                else:
                    labels_weight[bg_inds_unravel] = 1.0


            # ----------------------------------------
            # classification loss
            # ----------------------------------------
            labels = torch.tensor(labels, requires_grad=False)
            labels = labels.view(-1).type(torch.cuda.LongTensor)

            labels_weight = torch.tensor(labels_weight, requires_grad=False)
            labels_weight = labels_weight.view(-1).type(torch.cuda.FloatTensor)

            cls = cls.view(-1, 2)

            # cls loss
            active = labels_weight > 0
            loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
            loss_cls = (loss_cls * labels_weight[active]).mean()

            assert(labels_weight.sum() > 0)

            loss += loss_cls

            stats.append({'name': 'cls_p' + str(phase), 'val': loss_cls, 'format': '{:0.4f}', 'group': 'loss'})

            # ----------------------------------------
            # bbox regression loss
            # ----------------------------------------

            if np.sum(bbox_weights) > 0 and first_phase:

                bbox_weights = torch.tensor(bbox_weights, requires_grad=False).type(torch.cuda.FloatTensor).view(-1)

                active = bbox_weights > 0

                coords_2d_iou = coords_2d_iou.view(-1)
                stats.append({'name': 'iou_p' + str(phase), 'val': coords_2d_iou[active].mean(),
                              'format': '{:0.2f}', 'group': 'acc'})

                # use a 2d IoU based log loss
                if self.iou_2d_loss:
                    iou_2d_loss = -torch.log(coords_2d_iou[active])
                    iou_2d_loss = (iou_2d_loss * bbox_weights[active]).mean()

                    iou_2d_alpha = self.iou_2d_loss
                    iou_2d_loss *= iou_2d_alpha
                    loss += iou_2d_loss * iou_2d_alpha

                    stats.append({'name': 'iou_p' + str(phase), 'val': iou_2d_loss,
                                  'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
