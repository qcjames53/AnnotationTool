"""
This file is meant to contain functions which are
specific to region proposal networks.
"""

import matplotlib.pyplot as plt
import subprocess
import torch
import math
import re
import gc

from lib.util import *
from lib.core import *
from lib.math_3d import *
from lib.augmentations import *
from lib.nms.gpu_nms import gpu_nms
import torch.nn.functional as F

from copy import deepcopy


def generate_anchors(conf, imdb, cache_folder):
    """
    Generates the anchors according to the configuration and
    (optionally) based on the imdb properties.
    """

    use_el_z = 'use_el_z' in conf and conf.use_el_z
    decomp_alpha = 'decomp_alpha' in conf and conf.decomp_alpha
    has_vel = 'has_vel' in conf and conf.has_vel

    # use cache?
    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'anchors.pkl')):

        anchors = pickle_read(os.path.join(cache_folder, 'anchors.pkl'))

    # generate anchors
    else:

        anchors = np.zeros([len(conf.anchor_scales)*len(conf.anchor_ratios), 4], dtype=np.float32)

        aind = 0

        # compute simple anchors based on scale/ratios
        for scale in conf.anchor_scales:

            for ratio in conf.anchor_ratios:

                h = scale
                w = scale*ratio

                anchors[aind, 0:4] = anchor_center(w, h, conf.feat_stride)
                aind += 1

        # has 3d? then need to compute stats for each new dimension
        # presuming that anchors are initialized in "2d"
        if conf.has_3d:

            # compute the default stats for each anchor
            normalized_gts = None

            # check all images
            for imind, imobj in enumerate(imdb):

                # has ground truths?
                if len(imobj.gts) > 0:

                    scale = imobj.scale * conf.test_scale / imobj.imH

                    # determine ignores
                    igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                                   conf.min_gt_h, np.inf, scale)

                    # accumulate boxes
                    gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale for gt in imobj.gts]))
                    gts_val = gts_all[(rmvs == False) & (igns == False), :]

                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    if gts_val.shape[0] > 0:

                        # center all 2D ground truths
                        for gtind in range(0, gts_val.shape[0]):
                            w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                            h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                            gts_val[gtind, 0:4] = anchor_center(w, h, conf.feat_stride)

                    if gts_val.shape[0] > 0:
                        gt_info = np.ones([gts_val.shape[0], 100])*-1
                        gt_info[:, :(gts_val.shape[1] + gts_3d.shape[1])] = np.concatenate((gts_val, gts_3d), axis=1)
                        normalized_gts = gt_info if normalized_gts is None else np.vstack((normalized_gts, gt_info))

            # expand dimensions
            anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 5])), axis=1)

            if use_el_z:
                anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 1])), axis=1)

            if decomp_alpha:
                anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 2])), axis=1)

            if has_vel:
                anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 1])), axis=1)

            # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
            anchors_x3d = [[] for x in range(anchors.shape[0])]
            anchors_y3d = [[] for x in range(anchors.shape[0])]
            anchors_z3d = [[] for x in range(anchors.shape[0])]
            anchors_w3d = [[] for x in range(anchors.shape[0])]
            anchors_h3d = [[] for x in range(anchors.shape[0])]
            anchors_l3d = [[] for x in range(anchors.shape[0])]
            anchors_rotY = [[] for x in range(anchors.shape[0])]
            anchors_elv = [[] for x in range(anchors.shape[0])]
            anchors_sin = [[] for x in range(anchors.shape[0])]
            anchors_cos = [[] for x in range(anchors.shape[0])]
            anchors_vel = [[] for x in range(anchors.shape[0])]

            # find best matches for each ground truth
            ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])
            gt_target_ols = np.amax(ols, axis=0)
            gt_target_anchor = np.argmax(ols, axis=0)

            # assign each box to an anchor
            for gtind, gt in enumerate(normalized_gts):

                anum = gt_target_anchor[gtind]

                if gt_target_ols[gtind] > 0.2:
                    anchors_x3d[anum].append(gt[4])
                    anchors_y3d[anum].append(gt[5])
                    anchors_z3d[anum].append(gt[6])
                    anchors_w3d[anum].append(gt[7])
                    anchors_h3d[anum].append(gt[8])
                    anchors_l3d[anum].append(gt[9])
                    anchors_rotY[anum].append(gt[10])
                    anchors_elv[anum].append(gt[15])
                    anchors_sin[anum].append(gt[16])
                    anchors_cos[anum].append(gt[17])

                    if gt[20] >= 0:
                        anchors_vel[anum].append(gt[20])

            # compute global means
            anchors_x3d_gl = np.empty(0)
            anchors_y3d_gl = np.empty(0)
            anchors_z3d_gl = np.empty(0)
            anchors_w3d_gl = np.empty(0)
            anchors_h3d_gl = np.empty(0)
            anchors_l3d_gl = np.empty(0)
            anchors_rotY_gl = np.empty(0)
            anchors_elv_gl = np.empty(0)
            anchors_sin_gl = np.empty(0)
            anchors_cos_gl = np.empty(0)
            anchors_vel_gl = np.empty(0)

            # update anchors
            for aind in range(0, anchors.shape[0]):

                if len(np.array(anchors_z3d[aind])) > 0:

                    if conf.has_3d:

                        anchors_x3d_gl = np.hstack((anchors_x3d_gl, np.array(anchors_x3d[aind])))
                        anchors_y3d_gl = np.hstack((anchors_y3d_gl, np.array(anchors_y3d[aind])))
                        anchors_z3d_gl = np.hstack((anchors_z3d_gl, np.array(anchors_z3d[aind])))
                        anchors_w3d_gl = np.hstack((anchors_w3d_gl, np.array(anchors_w3d[aind])))
                        anchors_h3d_gl = np.hstack((anchors_h3d_gl, np.array(anchors_h3d[aind])))
                        anchors_l3d_gl = np.hstack((anchors_l3d_gl, np.array(anchors_l3d[aind])))
                        anchors_rotY_gl = np.hstack((anchors_rotY_gl, np.array(anchors_rotY[aind])))
                        anchors_elv_gl = np.hstack((anchors_elv_gl, np.array(anchors_elv[aind])))
                        anchors_sin_gl = np.hstack((anchors_sin_gl, np.array(anchors_sin[aind])))
                        anchors_cos_gl = np.hstack((anchors_cos_gl, np.array(anchors_cos[aind])))
                        anchors_vel_gl = np.hstack((anchors_vel_gl, np.array(anchors_vel[aind])))

                        anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                        anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                        anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                        anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                        anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))

                        if use_el_z:
                            anchors[aind, 9] = np.mean(np.array(anchors_elv[aind]))

                        if decomp_alpha:
                            anchors[aind, 9] = np.mean(np.array(anchors_sin[aind]))
                            anchors[aind, 10] = np.mean(np.array(anchors_cos[aind]))

                        if has_vel:
                            anchors[aind, 11] = np.mean(np.array(anchors_vel[aind]))

                else:
                    logging.info('WARNING: Non-used anchor #{} found. Removing this anchor.'.format(aind))
                    anchors[aind, :] = -1

        # remove non-used
        anchors = anchors[np.all(anchors == -1, axis=1) ==  False, :]

        # optionally cluster anchors
        if conf.cluster_anchors:
            anchors = cluster_anchors(conf.feat_stride, anchors, conf.test_scale, imdb, conf.lbls,
                                      conf.ilbls, conf.anchor_ratios, conf.min_gt_vis, conf.min_gt_h,
                                      conf.max_gt_h, conf.even_anchors, conf.expand_anchors)

        logging.info('Anchor info')

        for aind, anchor in enumerate(anchors):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            ar = w / h
            line = 'anchor {:2} w: {:6.2f}, h: {:6.2f}, ar: {:.2f}, z: {:5.2f}, w3d: {:.2f}, h3d: {:.2f}, l3d: {:.2f}, rot: {:5.2f}'.format(
                aind, w, h, ar, anchor[4], anchor[5], anchor[6], anchor[7], anchor[8]
            )

            if decomp_alpha:
                line += ', sin: {:6.2f}, cos: {:6.2f}'.format(anchor[9], anchor[10])

            if has_vel:
                line += ', vel: {:6.2f}'.format(anchor[11])

            logging.info(line)

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'anchors.pkl'), anchors)

    conf.anchors = anchors


def anchor_center(w, h, stride):
    """
    Centers an anchor based on a stride and the anchor shape (w, h).

    center ground truths with steps of half stride
    hence box 0 is centered at (7.5, 7.5) rather than (0, 0)
    for a feature stride of 16 px.
    """

    anchor = np.zeros([4], dtype=np.float32)

    anchor[0] = -w / 2 + (stride - 1) / 2
    anchor[1] = -h / 2 + (stride - 1) / 2
    anchor[2] = w / 2 + (stride - 1) / 2
    anchor[3] = h / 2 + (stride - 1) / 2

    return anchor


def cluster_anchors(feat_stride, anchors, test_scale, imdb, lbls, ilbls, anchor_ratios, min_gt_vis=0.99,
                    min_gt_h=0, max_gt_h=10e10, even_anchor_distribution=False, expand_anchors=False,
                    expand_stop_dt=0.0025):
    """
    Clusters the anchors based on the imdb boxes (in 2D and/or 3D).

    Generally, this method does a custom k-means clustering using 2D IoU
    as a distance metric.
    """

    normalized_gts = []

    # keep track if using 3d
    has_3d = False

    # check all images
    for imind, imobj in enumerate(imdb):

        # has ground truths?
        if len(imobj.gts) > 0:

            scale = imobj.scale * test_scale / imobj.imH

            # determine ignores
            igns, rmvs = determine_ignores(imobj.gts, lbls, ilbls, min_gt_vis, min_gt_h, np.inf, scale, use_trunc=True)

            # check for 3d box
            has_3d = 'bbox_3d' in imobj.gts[0]

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale for gt in imobj.gts]))
            gts_val = gts_all[(rmvs == False) & (igns == False), :]

            if has_3d:
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            if gts_val.shape[0] > 0:

                # center all 2D ground truths
                for gtind in range(0, gts_val.shape[0]):

                    w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                    h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                    gts_val[gtind, 0:4] = anchor_center(w, h, feat_stride)

            if gts_val.shape[0] > 0:

                # add normalized gts given 3d or 2d boxes
                if has_3d: normalized_gts += np.concatenate((gts_val, gts_3d), axis=1).tolist()
                else: normalized_gts += gts_val.tolist()

    # convert to np
    normalized_gts = np.array(normalized_gts)

    logging.info('starting clustering with {} ground truths'.format(normalized_gts.shape[0]))

    # sort by height
    sorted_inds = np.argsort((normalized_gts[:, 3] - normalized_gts[:, 1] + 1))
    normalized_gts = normalized_gts[sorted_inds, :]

    # init expand
    best_anchors = anchors
    expand_last_iou = 0
    expand_dif = 1
    best_met = 0
    best_cov = 0

    # init cluster
    max_rounds = 50
    round = 0
    last_iou = 0
    dif = 1

    while round < max_rounds and dif > -1000.0:

        # make empty arrays for each anchor
        anchors_h = [[] for x in range(anchors.shape[0])]
        anchors_w = [[] for x in range(anchors.shape[0])]

        if has_3d:

            # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
            anchors_z3d = [[] for x in range(anchors.shape[0])]
            anchors_w3d = [[] for x in range(anchors.shape[0])]
            anchors_h3d = [[] for x in range(anchors.shape[0])]
            anchors_l3d = [[] for x in range(anchors.shape[0])]
            anchors_rotY = [[] for x in range(anchors.shape[0])]

        round_ious = []
        round_zers = []
        round_mets = []

        # find best matches for each ground truth
        zers = np.abs(anchors[:, 4, np.newaxis] - normalized_gts[np.newaxis, :, 6])
        ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])

        metric = ols # - zers*0.01
        gt_target_anchor_learned = np.argmax(metric, axis=0)
        gt_target_anchor_matched = np.argmax(ols, axis=0)

        # to index manually metric[gt_target_anchor, range(metric.shape[1])]
        gt_mets = metric[gt_target_anchor_learned, range(metric.shape[1])]
        gt_ols = ols[gt_target_anchor_matched, range(metric.shape[1])]
        gt_zers = zers[gt_target_anchor_matched, range(metric.shape[1])]

        # assign each box to an anchor
        for gtind, gt in enumerate(normalized_gts):

            anum = gt_target_anchor_learned[gtind]

            w = gt[2] - gt[0] + 1
            h = gt[3] - gt[1] + 1

            anchors_h[anum].append(h)
            anchors_w[anum].append(w)

            if has_3d:
                anchors_z3d[anum].append(gt[6])
                anchors_w3d[anum].append(gt[7])
                anchors_h3d[anum].append(gt[8])
                anchors_l3d[anum].append(gt[9])
                anchors_rotY[anum].append(gt[10])

            # compute error by IoU matching
            round_ious.append(gt_ols[gtind])
            round_zers.append(gt_zers[gtind])
            round_mets.append(gt_mets[gtind])

        # compute errors
        cur_iou = np.mean(np.array(round_ious))
        cur_zer = np.mean(np.array(round_zers))
        cur_met = np.mean(np.array(round_mets))

        # update anchors
        for aind in range(0, anchors.shape[0]):

            # compute mean h/w
            if len(np.array(anchors_h[aind])) > 0:

                mean_h = np.mean(np.array(anchors_h[aind]))
                mean_w = np.mean(np.array(anchors_w[aind]))

                anchors[aind, 0:4] = anchor_center(mean_w, mean_h, feat_stride)

                if has_3d:
                    anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                    anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                    anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                    anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                    anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))

            else:
                raise ValueError('Non-used anchor #{} found'.format(aind))

        # store best configuration
        if cur_met > best_met:
            best_met = cur_iou
            best_anchors = anchors
            iou_cov = np.mean(np.array(round_ious) >= 0.5)
            zer_cov = np.mean(np.array(round_zers) <= 0.5)

            logging.info('clustering before update round {}, iou={:.4f}, z={:.4f}, met={:.4f}, iou_cov={:.2f}, z_cov={:.4f}'.format(round, cur_iou, cur_zer, cur_met, iou_cov, zer_cov))

        dif = cur_iou - last_iou
        last_iou = cur_iou

        round += 1

    return best_anchors


def compute_targets(gts_val, gts_ign, box_lbls, rois, fg_thresh, ign_thresh, bg_thresh_lo, bg_thresh_hi, best_thresh,
                    gts_3d=None, anchors=[], tracker=[], rois_3d=None, rois_3d_cen=None, decomp_trig_rot=False):
    """
    Computes the bbox targets of a set of rois and a set
    of ground truth boxes, provided various ignore
    settings in configuration
    """

    ols = None
    has_3d = gts_3d is not None
    use_el_z = anchors.shape[1] == 10
    decomp_alpha = anchors.shape[1] >= 11
    has_vel = anchors.shape[1] == 12

    # init transforms which respectively hold [dx, dy, dw, dh, label]
    # for labels bg=-1, ign=0, fg>=1
    transforms = np.zeros([len(rois), 5], dtype=np.float32)
    raw_gt = np.zeros([len(rois), 5], dtype=np.float32)

    # if 3d, then init other terms after
    if has_3d:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1]+use_el_z + decomp_alpha*2 + has_vel)], 'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1])], 'constant')

    if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

        if gts_ign.shape[0] > 0:

            # compute overlaps ign
            ols_ign = iou_ign(rois, gts_ign)
            ols_ign_max = np.amax(ols_ign, axis=1)

        else:
            ols_ign_max = np.zeros([rois.shape[0]], dtype=np.float32)

        if gts_val.shape[0] > 0:

            # compute overlaps valid
            ols = iou(rois, gts_val)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= best_thresh]
            gt_best_ols = gt_best_ols[gt_best_ols >= best_thresh]

            fg_inds = np.flatnonzero(ols_max >= fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)

            target_rois = gts_val[targets[fg_inds], :]
            src_rois = rois[fg_inds, :]

            if len(fg_inds) > 0:

                # compute 2d transform
                transforms[fg_inds, 0:4] = bbox_transform(src_rois, target_rois)

                raw_gt[fg_inds, 0:4] = target_rois

                if has_3d:

                    tracker = tracker.astype(np.int64)
                    if rois_3d is None:
                        src_3d = anchors[tracker[fg_inds], 4:]
                    else:
                        src_3d = rois_3d[fg_inds, 4:]
                    target_3d = gts_3d[targets[fg_inds]]

                    raw_gt[fg_inds, 5:] = target_3d

                    if rois_3d_cen is None:

                        # compute 3d transform
                        transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d, use_el_z=use_el_z, decomp_alpha=decomp_alpha, has_vel=has_vel, decomp_trig_rot=decomp_trig_rot)

                    else:
                        transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d, use_el_z=use_el_z, decomp_alpha=decomp_alpha, has_vel=has_vel, decomp_trig_rot=decomp_trig_rot, rois_3d_cen=rois_3d_cen[fg_inds])


                # store labels
                transforms[fg_inds, 4] = [box_lbls[x] for x in targets[fg_inds]]
                assert (all(transforms[fg_inds, 4] >= 1))

        else:

            ols_max = np.zeros(rois.shape[0], dtype=int)
            fg_inds = np.empty(shape=[0])
            gt_best_rois = np.empty(shape=[0])

        # determine ignores
        ign_inds = np.flatnonzero(ols_ign_max >= ign_thresh)

        # determine background
        bg_inds = np.flatnonzero((ols_max >= bg_thresh_lo) & (ols_max < bg_thresh_hi))

        # subtract fg and igns from background
        bg_inds = np.setdiff1d(bg_inds, ign_inds)
        bg_inds = np.setdiff1d(bg_inds, fg_inds)
        bg_inds = np.setdiff1d(bg_inds, gt_best_rois)

        # mark background
        transforms[bg_inds, 4] = -1

    else:

        # all background
        transforms[:, 4] = -1


    return transforms, ols, raw_gt


def hill_climb(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_z_init=0, step_r_init=0, z_lim=0, r_lim=0, min_ol_dif=0.0, alpha=False):

    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid: return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):

        if step_z > z_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d - step_z, w3d, h3d, l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d + step_z, w3d, h3d, l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return z2d, ry3d, verts_best


def clsInd2Name(lbls, ind):
    """
    Converts a cls ind to string name
    """

    if ind>=0 and ind<len(lbls):
        return lbls[ind]
    else:
        raise ValueError('unknown class')


def clsName2Ind(lbls, cls):
    """
    Converts a cls name to an ind
    """
    if cls in lbls:
        return lbls.index(cls) + 1
    else:
        raise ValueError('unknown class')


def compute_bbox_stats(conf, imdb, cache_folder=''):
    """
    Computes the mean and standard deviation for each regression
    parameter (usually pertaining to [dx, dy, sw, sh] but sometimes
    for 3d parameters too).

    Once these stats are known we normalize the regression targets
    to have 0 mean and 1 variance, to hypothetically ease training.
    """

    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'bbox_means.pkl')) \
            and os.path.exists(os.path.join(cache_folder, 'bbox_stds.pkl')):

        means = pickle_read(os.path.join(cache_folder, 'bbox_means.pkl'))
        stds = pickle_read(os.path.join(cache_folder, 'bbox_stds.pkl'))

    else:

        if ('use_el_z' in conf) and conf.use_el_z:
            squared_sums = np.zeros([1, 12], dtype=np.float128)
            sums = np.zeros([1, 12], dtype=np.float128)
        elif ('has_vel' in conf) and conf.has_vel:
            squared_sums = np.zeros([1, 14], dtype=np.float128)
            sums = np.zeros([1, 14], dtype=np.float128)
        elif ('decomp_alpha' in conf) and conf.decomp_alpha:
            squared_sums = np.zeros([1, 13], dtype=np.float128)
            sums = np.zeros([1, 13], dtype=np.float128)
        elif conf.has_3d:
            squared_sums = np.zeros([1, 11], dtype=np.float128)
            sums = np.zeros([1, 11], dtype=np.float128)
        else:
            squared_sums = np.zeros([1, 4], dtype=np.float128)
            sums = np.zeros([1, 4], dtype=np.float128)

        class_counts = np.zeros([1], dtype=np.float128) + 1e-10
        class_counts_vel = np.zeros([1], dtype=np.float128) + 1e-10

        # compute the mean first
        logging.info('Computing bbox regression mean..')

        for imind, imobj in enumerate(imdb):

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                               conf.min_gt_h, np.inf, scale_factor, use_trunc=True)

                # accumulate boxes
                gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _= compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, gts_3d=gts_3d,
                                                    anchors=conf.anchors, tracker=rois[:, 4], decomp_trig_rot=conf.decomp_trig_rot)
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, decomp_trig_rot=conf.decomp_trig_rot)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if ('use_el_z' in conf) and conf.use_el_z:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:13], axis=0)
                    elif ('has_vel' in conf) and conf.has_vel:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:13] += np.sum(transforms[gt_inds, 5:14], axis=0)

                        valid_vel = transforms[gt_inds, 14] > (-np.inf)
                        sums[:, 13] += transforms[gt_inds, 14][valid_vel].sum()
                        class_counts_vel += valid_vel.sum()

                    elif ('decomp_alpha' in conf) and conf.decomp_alpha:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:14], axis=0)
                    elif conf.has_3d:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:12], axis=0)
                    else:
                        sums += np.sum(transforms[gt_inds, 0:4], axis=0)

                    class_counts += len(gt_inds)

        means = sums/class_counts

        if ('has_vel' in conf) and conf.has_vel:
            means[:, 13] = sums[:, 13] / class_counts_vel

        logging.info('Computing bbox regression stds..')

        for imobj in imdb:

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis, conf.min_gt_h, np.inf, scale_factor)

                # accumulate boxes
                gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, gts_3d=gts_3d,
                                                    anchors=conf.anchors, tracker=rois[:, 4], decomp_trig_rot=conf.decomp_trig_rot)
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, decomp_trig_rot=conf.decomp_trig_rot)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if ('use_el_z' in conf) and conf.use_el_z:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:13] - means[:, 4:], 2), axis=0)
                    elif ('has_vel' in conf) and conf.has_vel:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:13] += np.sum(np.power(transforms[gt_inds, 5:14] - means[:, 4:13], 2), axis=0)
                        valid_vel = transforms[gt_inds, 14] > (-np.inf)
                        squared_sums[:, 13] += np.power(transforms[gt_inds, 14][valid_vel] - means[:, 13], 2).sum()
                    elif ('decomp_alpha' in conf) and conf.decomp_alpha:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:14] - means[:, 4:], 2), axis=0)
                    elif conf.has_3d:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:12] - means[:, 4:], 2), axis=0)

                    else:
                        squared_sums += np.sum(np.power(transforms[gt_inds, 0:4] - means, 2), axis=0)

        stds = np.sqrt((squared_sums/class_counts))

        if ('has_vel' in conf) and conf.has_vel:
            stds[:, 13] = np.sqrt((squared_sums[:, 13]/class_counts_vel))

        means = means.astype(float)
        stds = stds.astype(float)

        logging.info('used {:d} boxes with avg std {:.4f}'.format(int(class_counts[0]), np.mean(stds)))

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'bbox_means.pkl'), means)
            pickle_write(os.path.join(cache_folder, 'bbox_stds.pkl'), stds)

    conf.bbox_means = means
    conf.bbox_stds = stds

def compute_plane_stats(conf, imdb):

    squared_sums = np.zeros([1, 4], dtype=np.float128)
    sums = np.zeros([1, 4], dtype=np.float128)

    class_counts = np.zeros([1], dtype=np.float128) + 1e-10

    # compute the mean first
    logging.info('Computing plane regression mean..')

    for imind, imobj in enumerate(imdb):

        if imobj.plane_gt is not None:

            sums[:, 0:4] += imobj.plane_gt
            class_counts += 1

    means = sums/class_counts

    logging.info('Computing plane regression stds..')

    for imind, imobj in enumerate(imdb):

        if imobj.plane_gt is not None:

            squared_sums[:, 0:4] += np.power((imobj.plane_gt - means[0, 0:4]), 2)

    stds = np.sqrt((squared_sums/class_counts))

    means = means.astype(float)
    stds = stds.astype(float)

    logging.info('mean plane: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(means[0,0], means[0, 1], means[0, 2], means[0, 3]))
    logging.info('stds plane: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(stds[0, 0], stds[0, 1], stds[0, 2], stds[0, 3]))

    conf.plane_means = means
    conf.plane_stds = stds


def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    """

    bsize = input.shape[0]
    csize = input.shape[1]

    return input.permute(0, 2, 3, 1).contiguous().view(bsize, -1, csize)


def unflatten_tensor(input, feat_size, anchors):
    """
    Un-flattens and un-permutes a tensor from size
    [B x (W x H) x C] --> [B x C x W x H]
    """

    bsize = input.shape[0]

    if len(input.shape) >= 3: csize = input.shape[2]
    else: csize = 1

    input = input.view(bsize, feat_size[0] * anchors.shape[0], feat_size[1], csize)
    input = input.permute(0, 3, 1, 2).contiguous()

    return input


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """


    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
        y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
        z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        #corners_2D = np.zeros([3, corners_3d.shape[1]])
        #for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def project_3d_corners(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    '''
    order of vertices
    0  upper back right
    1  upper front right
    2  bottom front right
    3  bottom front left
    4  upper front left
    5  upper back left
    6  bottom back left
    7  bottom back right
    
    bot_inds = np.array([2,3,6,7])
    top_inds = np.array([0,1,4,5])
    '''

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    return corners_2D, corners_3D_1


def bbCoords2XYWH(box):
    """
    Convert from [x1, y1, x2, y2] to [x,y,w,h]
    """

    if box.shape[0] == 0: return np.empty([0, 4], dtype=float)

    box[:, 2] -= box[:, 0] + 1
    box[:, 3] -= box[:, 1] + 1

    return box


def bbXYWH2Coords(box):
    """
    Convert from [x,y,w,h] to [x1, y1, x2, y2]
    """

    if box.shape[0] == 0: return np.empty([0,4], dtype=float)

    box[:, 2] += box[:, 0] - 1
    box[:, 3] += box[:, 1] - 1

    return box


def bbox_transform_3d(ex_rois_2d, ex_rois_3d, gt_rois, use_el_z=False, decomp_alpha=False, has_vel=False, rois_3d_cen=None, decomp_trig_rot=False):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois_2d[:, 2] - ex_rois_2d[:, 0] + 1.0
    ex_heights = ex_rois_2d[:, 3] - ex_rois_2d[:, 1] + 1.0

    if rois_3d_cen is None:
        ex_ctr_x = ex_rois_2d[:, 0] + 0.5 * (ex_widths)
        ex_ctr_y = ex_rois_2d[:, 1] + 0.5 * (ex_heights)
    else:
        ex_ctr_x = rois_3d_cen[:, 0]
        ex_ctr_y = rois_3d_cen[:, 1]

    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    delta_z = gt_rois[:, 2] - ex_rois_3d[:, 0]
    scale_w = np.log(gt_rois[:, 3] / ex_rois_3d[:, 1])
    scale_h = np.log(gt_rois[:, 4] / ex_rois_3d[:, 2])
    scale_l = np.log(gt_rois[:, 5] / ex_rois_3d[:, 3])
    deltaRotY = gt_rois[:, 6] - ex_rois_3d[:, 4]

    if decomp_trig_rot:
        deltaRotY = np.sin(deltaRotY)

    if use_el_z:
        delta_elv = gt_rois[:, 11] - ex_rois_3d[:, 5]
        targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY, delta_elv)).transpose()
    elif decomp_alpha:
        delta_sin = np.sin(gt_rois[:, 12] - ex_rois_3d[:, 5])
        delta_cos = np.cos(gt_rois[:, 13] - ex_rois_3d[:, 6] + math.pi/2)

        if has_vel:
            delta_vel = np.ones(delta_sin.shape)*(-np.inf) if gt_rois.shape[1] != 17 else gt_rois[:, 16] - ex_rois_3d[:, 7]
            targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY, delta_sin, delta_cos, delta_vel)).transpose()
        else:
            targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY, delta_sin, delta_cos)).transpose()
    else:
        targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY)).transpose()

    targets = np.hstack((targets, gt_rois[:, 7:]))


    return targets


def bbox_transform(ex_rois, gt_rois):
    """
    Compute the bbox target transforms in 2D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    # boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths)
    ctr_y = boxes[:, 1] + 0.5 * (heights)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros(deltas.shape)

    # x1, y1, x2, y2
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w)
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h)
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w) - 1
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h) - 1

    return pred_boxes


def determine_ignores(gts, lbls, ilbls, min_gt_vis=0.99, min_gt_h=0, max_gt_h=10e10, scale_factor=1, use_trunc=False):
    """
    Given various configuration settings, determine which ground truths
    are ignored and which are relevant.
    """

    igns = np.zeros([len(gts)], dtype=bool)
    rmvs = np.zeros([len(gts)], dtype=bool)

    for gtind, gt in enumerate(gts):

        ign = gt.ign
        ign |= gt.visibility < min_gt_vis
        ign |= gt.bbox_full[3] * scale_factor < min_gt_h
        ign |= gt.bbox_full[3] * scale_factor > max_gt_h
        ign |= gt.cls in ilbls

        if use_trunc:
            ign |= gt.trunc > max(1 - min_gt_vis, 0)

        rmv = not gt.cls in (lbls + ilbls)

        igns[gtind] = ign
        rmvs[gtind] = rmv

    return igns, rmvs


def locate_anchors(anchors, feat_size, stride, convert_tensor=False):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across
        stride (int): stride of a network
        convert_tensor (bool, optional): whether to return a torch tensor, otherwise ndarray [default=False]

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # torch and numpy MAY have different calls for reshaping, although
    # it is not very important which is used as long as it is CONSISTENT
    if convert_tensor:

        # important to unroll according to pytorch
        shift_x1 = torch.from_numpy(shift_x1).view(1, stack_size, feat_size[1])
        shift_y1 = torch.from_numpy(shift_y1).view(1, stack_size, feat_size[1])
        shift_x2 = torch.from_numpy(shift_x2).view(1, stack_size, feat_size[1])
        shift_y2 = torch.from_numpy(shift_y2).view(1, stack_size, feat_size[1])
        anchor_tracker = torch.from_numpy(anchor_tracker).view(1, stack_size, feat_size[1])

        shift_x1.requires_grad = False
        shift_y1.requires_grad = False
        shift_x2.requires_grad = False
        shift_y2.requires_grad = False
        anchor_tracker.requires_grad = False

        shift_x1 = shift_x1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y1 = shift_y1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_x2 = shift_x2.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y2 = shift_y2.permute(1, 2, 0).contiguous().view(-1, 1)
        anchor_tracker = anchor_tracker.permute(1, 2, 0).contiguous().view(-1, 1)

        rois = torch.cat((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    else:

        shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)

        rois = np.concatenate((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois


def calc_output_size(res, stride):
    """
    Approximate the output size of a network

    Args:
        res (ndarray): input resolution
        stride (int): stride of a network

    Returns:
         ndarray: output resolution
    """

    return np.ceil(np.array(res)/stride).astype(int)


def im_detect_2d(im, net, rpn_conf, preprocess, gpu=0, synced=False):
    """
    Object detection in 2D
    """

    imH_orig = im.shape[0]
    imW_orig = im.shape[1]

    im = preprocess(im)

    # move to GPU
    im = torch.from_numpy(im).cuda()

    imH = im.shape[2]
    imW = im.shape[3]

    scale_factor = imH / imH_orig

    cls, prob, bbox_2d = net(im)

    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]
    feat_size = [int(bbox_2d[0].shape[2] / num_anchors), bbox_2d[0].shape[3]]

    # flatten everything
    prob = flatten_tensor(prob)

    bbox_x = flatten_tensor(bbox_2d[0])
    bbox_y = flatten_tensor(bbox_2d[1])
    bbox_w = flatten_tensor(bbox_2d[2])
    bbox_h = flatten_tensor(bbox_2d[3])

    rois = locate_anchors(rpn_conf.anchors, feat_size, rpn_conf.feat_stride, convert_tensor=True)
    rois = rois.type(torch.cuda.FloatTensor)

    # compile deltas pred
    deltas_2d = torch.cat((bbox_x[0, :, :], bbox_y[0, :, :], bbox_w[0, :, :], bbox_h[0, :, :]), dim=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

    # detach onto cpu
    coords_2d = coords_2d.cpu().detach().numpy()
    prob = prob[0, :, :].cpu().detach().numpy()

    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]

    if synced:

        # nms
        keep_inds = gpu_nms(aboxes.astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    else:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]

        # nms
        keep_inds = gpu_nms(aboxes.astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # scale coords
    aboxes[:, 0:4] /= scale_factor

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    return aboxes


def im_detect_3d_forecast(ims, net, rpn_conf, scale_factor, p2, gpu=0, synced=False, return_poses=False):
    """
    Object detection in 3D
    """

    p2_inv = np.linalg.inv(p2)[np.newaxis, :, :]
    p2 = p2[np.newaxis, :, :]

    if ('is_kalman' in rpn_conf) and rpn_conf.is_kalman:
        si_shots, tr_shots, poses, feat_size, rois, rois_3d, rois_3d_cen = net(ims[np.newaxis, :, :, :], p2, p2_inv, [scale_factor])

        prob_raw = si_shots[0][1]
        bbox_2d = si_shots[0][2]
        bbox_3d = si_shots[0][3][:, :, :]

    else:
        si_shots, eg_shots, tr_shots, poses, feat_size, rois, rois_3d, rois_3d_cen = net(ims[np.newaxis, :, :, :], p2, p2_inv, [scale_factor])

        prob_raw = torch.cat((si_shots[0][1], eg_shots[0][1]), dim=0)
        bbox_2d = torch.cat((si_shots[0][2], eg_shots[0][2]), dim=0)
        bbox_3d = torch.cat((si_shots[0][3][:, :, :11], eg_shots[0][3][:, :, :11], ), dim=0)

    #flow10 = flow10[0, :, :].detach().cpu().numpy() * 10e-3

    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]

    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d_raw = bbox_3d[:, :, 0]
    bbox_y3d_raw = bbox_3d[:, :, 1]
    bbox_z3d_raw = bbox_3d[:, :, 2]
    bbox_w3d_raw = bbox_3d[:, :, 3]
    bbox_h3d_raw = bbox_3d[:, :, 4]
    bbox_l3d_raw = bbox_3d[:, :, 5]
    #bbox_ry3d_raw = bbox_3d[:, :, 6]

    if ('has_un' in rpn_conf) and rpn_conf.has_un:
        bbox_un = bbox_3d[:, :, 7]

    elif ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:
        bbox_rsin_raw = bbox_3d[:, :, 6]
        bbox_rcos_raw = bbox_3d[:, :, 7]
        bbox_axis_raw = bbox_3d[:, :, 8]
        bbox_head_raw = bbox_3d[:, :, 9]
        bbox_rsin_raw = bbox_rsin_raw * rpn_conf.bbox_stds[:, 11][0] + rpn_conf.bbox_means[:, 11][0]
        bbox_rcos_raw = bbox_rcos_raw * rpn_conf.bbox_stds[:, 12][0] + rpn_conf.bbox_means[:, 12][0]

    if ('has_vel' in rpn_conf) and rpn_conf.has_vel:
        if si_shots[0][3].shape[2] == 20:
            bbox_vel_raw = si_shots[0][3][:, :, 19]
        else:
            bbox_vel_raw = si_shots[0][3][:, :, 10]

        bbox_vel_raw = bbox_vel_raw * rpn_conf.bbox_stds[:, 13][0] + rpn_conf.bbox_means[:, 13][0]

    # detransform 3d
    bbox_x3d_raw = bbox_x3d_raw * rpn_conf.bbox_stds[:, 4][0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d_raw = bbox_y3d_raw * rpn_conf.bbox_stds[:, 5][0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d_raw = bbox_z3d_raw * rpn_conf.bbox_stds[:, 6][0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d_raw = bbox_w3d_raw * rpn_conf.bbox_stds[:, 7][0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d_raw = bbox_h3d_raw * rpn_conf.bbox_stds[:, 8][0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d_raw = bbox_l3d_raw * rpn_conf.bbox_stds[:, 9][0] + rpn_conf.bbox_means[:, 9][0]
    #bbox_ry3d_raw = bbox_ry3d_raw * rpn_conf.bbox_stds[:, 10][0] + rpn_conf.bbox_means[:, 10][0]

    # find 3d source
    tracker_raw = rois[:, :, 4].cpu().detach().numpy().astype(np.int64)
    src_3d = rois_3d[:, :, 4:]

    # compute 3d transform
    widths = rois[:, :, 2] - rois[:, :, 0] + 1.0
    heights = rois[:, :, 3] - rois[:, :, 1] + 1.0
    ctr_x = rois[:, :, 0] + 0.5 * widths
    ctr_y = rois[:, :, 1] + 0.5 * heights

    aboxes_all = []

    for bind in range(prob_raw.shape[0]):

        bbox_x3d = bbox_x3d_raw[bind, :] * widths[0] + rois_3d_cen[0, :, 0]
        bbox_y3d = bbox_y3d_raw[bind, :] * heights[0] + rois_3d_cen[0, :, 1]
        bbox_z3d = src_3d[0, :, 0] + bbox_z3d_raw[bind, :]
        bbox_w3d = torch.exp(bbox_w3d_raw[bind, :]) * src_3d[0, :, 1]
        bbox_h3d = torch.exp(bbox_h3d_raw[bind, :]) * src_3d[0, :, 2]
        bbox_l3d = torch.exp(bbox_l3d_raw[bind, :]) * src_3d[0, :, 3]
        #bbox_ry3d = src_3d[0, :, 4] + bbox_ry3d_raw[bind, :]

        if ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:
            bbox_rsin = src_3d[0, :, 5] + torch.asin(bbox_rsin_raw[bind, :].clamp(min=-1, max=1))
            bbox_rcos = src_3d[0, :, 6] + torch.acos(bbox_rcos_raw[bind, :].clamp(min=-1, max=1)) - math.pi / 2
            bbox_axis_sin_mask = bbox_axis_raw[bind, :] >= 0.5
            bbox_head_pos_mask = bbox_head_raw[bind, :] >= 0.5

            bbox_ry3d = bbox_rcos
            bbox_ry3d[bbox_axis_sin_mask] = bbox_rsin[bbox_axis_sin_mask]
            bbox_ry3d[bbox_head_pos_mask] = bbox_ry3d[bbox_head_pos_mask] + math.pi

        if ('has_vel' in rpn_conf) and rpn_conf.has_vel and bind == 0:
            bbox_vel_dn = (src_3d[0, :, 7] + bbox_vel_raw[bind, :]).clamp(min=0)

        # bundle
        coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]], bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]), dim=1)

        if ('has_un' in rpn_conf) and rpn_conf.has_un:
            coords_3d = torch.cat((coords_3d, bbox_un.t()), dim=1)

        if ('has_vel' in rpn_conf) and rpn_conf.has_vel and bind == 0:
            coords_3d = torch.cat((coords_3d, bbox_vel_dn[:bbox_x3d.shape[0], np.newaxis]), dim=1)

        # compile deltas pred
        deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis], bbox_y[bind, :, np.newaxis], bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)
        coords_2d = bbox_transform_inv(rois[0], deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

        # detach onto cpu
        coords_2d = coords_2d.cpu().detach().numpy()
        coords_3d = coords_3d.cpu().detach().numpy()
        prob = prob_raw[bind, :, :].cpu().detach().numpy()

        # scale coords
        coords_2d[:, 0:4] /= scale_factor
        coords_3d[:, 0:2] /= scale_factor

        cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
        scores = np.amax(prob[:, 1:], axis=1)

        aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

        sorted_inds = (-aboxes[:, 4]).argsort()
        original_inds = (sorted_inds).argsort()
        aboxes = aboxes[sorted_inds, :]
        coords_3d = coords_3d[sorted_inds, :]
        cls_pred = cls_pred[sorted_inds]
        tracker = tracker_raw[0, sorted_inds]

        #if bind==2:
        #    flow10 = flow10[sorted_inds, :]

        if synced:

            # nms
            keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

            # convert to bool
            keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
            keep[keep_inds, :] = True

            # stack the keep array,
            # sync to the original order
            aboxes = np.hstack((aboxes, keep))
            aboxes[original_inds, :]

        else:

            # pre-nms
            cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
            tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
            aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
            coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

            # nms
            keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

            # stack cls prediction
            aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

            # suppress boxes
            aboxes = aboxes[keep_inds, :]

            #if bind==2:
            #    flow10 = flow10[0:min(rpn_conf.nms_topN_pre, flow10.shape[0])]
            #    flow10 = flow10[keep_inds, :]


        aboxes_all.append(aboxes)

    if ('is_kalman' in rpn_conf) and rpn_conf.is_kalman:
        tracks = tr_shots[0]
        if tracks is None or tracks[0] is None or tracks[0].Xs.shape[0] == 0:
            aboxes_all.append([])
        else:
            bbox_2d = tracks[0].box2ds.detach().cpu().numpy()
            Xs = tracks[0].Xs.detach().cpu().numpy()

            # apply head
            Xs[Xs[:, 7] >= 0.5, 6] += math.pi
            Cs = tracks[0].Cs.detach().cpu().numpy()
            Cs = np.array([a.diagonal() for a in Cs])

            aboxes_all.append(np.concatenate((bbox_2d, Xs, Cs), axis=1))

    if return_poses:
        return aboxes_all, poses[2].detach().cpu().numpy()
    else:
        return aboxes_all


def im_detect_3d(im, net, rpn_conf, preprocess, p2, gpu=0, synced=False, return_base=False, depth_map=None):
    """
    Object detection in 3D
    """

    imH_orig = im.shape[0]
    imW_orig = im.shape[1]

    p2_inv = np.linalg.inv(p2)

    p2_a = p2[0, 0].item()
    p2_b = p2[0, 2].item()
    p2_c = p2[0, 3].item()
    p2_d = p2[1, 1].item()
    p2_e = p2[1, 2].item()
    p2_f = p2[1, 3].item()
    p2_h = p2[2, 3].item()

    im = preprocess(im)

    # move to GPU
    im = torch.from_numpy(im[np.newaxis, :, :, :]).cuda()

    imH = im.shape[2]
    imW = im.shape[3]

    scale_factor = imH / imH_orig

    if ('use_el_z' in rpn_conf) and rpn_conf.use_el_z:
        p2_inv = np.linalg.inv(p2)[np.newaxis, :, :]
        p2 = p2[np.newaxis, :, :]

        cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im, p2, p2_inv, [scale_factor])
    elif ('use_gp' in rpn_conf) and rpn_conf.use_gp:
        cls, prob, bbox_2d, bbox_3d, plane, feat_size, rois = net(im)
    elif return_base:
        cls, prob, bbox_2d, bbox_3d, feat_size, rois, base = net(im, return_base=return_base)
    elif depth_map is not None:
        cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im, depth_map)
    else:
        cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im)

    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]

    if not ('infer_2d_from_3d' in rpn_conf) or not (rpn_conf.infer_2d_from_3d):
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
    #bbox_ry3d = bbox_3d[:, :, 6]

    if ('use_el_z' in rpn_conf) and rpn_conf.use_el_z:
        bbox_elv = bbox_3d[:, :, 7]
        bbox_un = bbox_3d[:, :, 8]

    elif ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:
        bbox_rsin = bbox_3d[:, :, 6]
        bbox_rcos = bbox_3d[:, :, 7]
        bbox_axis = bbox_3d[:, :, 8]
        bbox_head = bbox_3d[:, :, 9]
        bbox_rsin = bbox_rsin * rpn_conf.bbox_stds[:, 11][0] + rpn_conf.bbox_means[:, 11][0]
        bbox_rcos = bbox_rcos * rpn_conf.bbox_stds[:, 12][0] + rpn_conf.bbox_means[:, 12][0]

    if ('has_un' in rpn_conf) and rpn_conf.has_un and ('ind_un' in rpn_conf) and rpn_conf.ind_un:
        bbox_un = bbox_3d[:, :, 7+2]
    elif ('has_un' in rpn_conf) and rpn_conf.has_un and (('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha):
        bbox_un = bbox_3d[:, :, 11]
    elif ('has_un' in rpn_conf) and rpn_conf.has_un:
        bbox_un = bbox_3d[:, :, 7]

    # detransform 3d
    bbox_x3d = bbox_x3d * rpn_conf.bbox_stds[:, 4][0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * rpn_conf.bbox_stds[:, 5][0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * rpn_conf.bbox_stds[:, 6][0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * rpn_conf.bbox_stds[:, 7][0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * rpn_conf.bbox_stds[:, 8][0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * rpn_conf.bbox_stds[:, 9][0] + rpn_conf.bbox_means[:, 9][0]
    #bbox_ry3d = bbox_ry3d * rpn_conf.bbox_stds[:, 10][0] + rpn_conf.bbox_means[:, 10][0]

    # find 3d source
    tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)

    #tracker_sca = rois_sca[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d_sca = torch.from_numpy(rpn_conf.anchors[tracker_sca, 4:]).cuda().type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y

    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = torch.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = torch.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = torch.exp(bbox_l3d[0, :]) * src_3d[:, 3]

    #if ('decomp_trig_rot' in rpn_conf) and rpn_conf.decomp_trig_rot:
    #    bbox_ry3d = src_3d[:, 4] + torch.asin(bbox_ry3d[0, :].clamp(min=-1, max=1))
    #else:
    #    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    if ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:
        bbox_rsin = src_3d[:, 5] + torch.asin(bbox_rsin[0, :].clamp(min=-1, max=1))
        bbox_rcos = src_3d[:, 6] + torch.acos(bbox_rcos[0, :].clamp(min=-1, max=1)) - math.pi / 2
        bbox_axis_sin_mask = bbox_axis[0, :] >= 0.5
        bbox_head_pos_mask = bbox_head[0, :] >= 0.5

        bbox_ry3d = bbox_rcos
        bbox_ry3d[bbox_axis_sin_mask] = bbox_rsin[bbox_axis_sin_mask]

        bbox_ry3d[bbox_head_pos_mask] = bbox_ry3d[bbox_head_pos_mask] + math.pi

        a = 1

    # using elevation?
    if ('use_el' in rpn_conf) and rpn_conf.use_el:
        # compute elevation
        y3d = bbox_z3d - bbox_h3d * 0.5
        # solve for z
        #bbox_z3d = ((y3d * p2_d + p2_f - p2_h * bbox_y3d/scale_factor) / (bbox_y3d/scale_factor - p2_e)) + p2_h
        bbox_z3d = (y3d * p2_d - p2_e * p2_h + p2_f)/((bbox_y3d/scale_factor) - p2_e)
        bbox_z3d = bbox_z3d.clamp(min=-5, max=65)

    # need to convert elevation into 2D pixel
    if ('use_el_z' in rpn_conf) and rpn_conf.use_el_z:

        # center the elevation
        y3d = bbox_elv[0, :] - bbox_h3d * 0.5

        # solve for z!
        z3d_elv = (y3d * p2_d - p2_e * p2_h + p2_f) / ((bbox_y3d / scale_factor) - p2_e)
        z3d_elv = z3d_elv.clamp(min=1, max=65)

        # store
        bbox_z3d = z3d_elv

    # bundle
    coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]], bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]), dim=1)

    if ('use_el_z' in rpn_conf) and rpn_conf.use_el_z:
        coords_3d = torch.cat((coords_3d, bbox_un.t()), dim=1)
    elif ('has_un' in rpn_conf) and rpn_conf.has_un:
        coords_3d = torch.cat((coords_3d, bbox_un.t()), dim=1)


    if not ('infer_2d_from_3d' in rpn_conf) or not (rpn_conf.infer_2d_from_3d):

        # compile deltas pred
        deltas_2d = torch.cat((bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis], bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]), dim=1)
        coords_2d = bbox_transform_inv(rois, deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

        # detach onto cpu
        coords_2d = coords_2d.cpu().detach().numpy()
        coords_3d = coords_3d.cpu().detach().numpy()
        prob = prob[0, :, :].cpu().detach().numpy()

        # scale coords
        coords_2d[:, 0:4] /= scale_factor
        coords_3d[:, 0:2] /= scale_factor

        cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
        scores = np.amax(prob[:, 1:], axis=1)

    else:
        coords_3d = coords_3d.cpu().detach().numpy()
        prob = prob[0, :, :].cpu().detach().numpy()

        # scale coords
        coords_3d[:, 0:2] /= scale_factor

        cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
        scores = np.amax(prob[:, 1:], axis=1)

        fg_mask = scores > rpn_conf.score_thres
        fg_inds = np.flatnonzero(fg_mask)

        coords_3d = coords_3d[fg_inds, :]
        prob = prob[fg_inds, :]
        scores = scores[fg_inds]
        cls_pred = cls_pred[fg_inds]

        # get info
        x2d = coords_3d[:, 0]
        y2d = coords_3d[:, 1]
        z2d = coords_3d[:, 2]
        w3d = coords_3d[:, 3]
        h3d = coords_3d[:, 4]
        l3d = coords_3d[:, 5]
        alp = coords_3d[:, 6]

        coords_3d_proj = p2_inv.dot(np.vstack((x2d*z2d, y2d*z2d, z2d, np.ones(x2d.shape))))
        x3d = coords_3d_proj[0]
        y3d = coords_3d_proj[1]
        z3d = coords_3d_proj[2]

        ry3d = convertAlpha2Rot(alp, z3d, x3d)

        coords_2d, ign = get_2D_from_3D(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    if synced and aboxes.shape[0] > 0:

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    elif aboxes.shape[0] > 0:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    if ('use_gp' in rpn_conf) and rpn_conf.use_gp:
        return aboxes, plane
    elif return_base:
        return aboxes, base
    else:
        return aboxes


def get_2D_from_3D(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY):

    if type(cx3d) == torch.Tensor:
        cx3d = cx3d.detach()
        cy3d = cy3d.detach()
        cz3d = cz3d.detach()
        w3d = w3d.detach()
        h3d = h3d.detach()
        l3d = l3d.detach()
        rotY = rotY.detach()

    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    if type(cx3d) == np.ndarray:
        ign = np.any(corners_3d[:, 2, :] <= 0, axis=1)
        x = verts3d[:, 0, :].min(axis=1)
        y = verts3d[:, 1, :].min(axis=1)
        x2 = verts3d[:, 0, :].max(axis=1)
        y2 = verts3d[:, 1, :].max(axis=1)

        return np.vstack((x, y, x2, y2)).T, ign

    if type(cx3d) == torch.Tensor:
        ign = torch.any(corners_3d[:, 2, :] <= 0, dim=1)
        x = verts3d[:, 0, :].min(dim=1)[0]
        y = verts3d[:, 1, :].min(dim=1)[0]
        x2 = verts3d[:, 0, :].max(dim=1)[0]
        y2 = verts3d[:, 1, :].max(dim=1)[0]

        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1), ign

    else:
        # any boxes behind camera plane?
        ign = np.any(corners_3d[2, :] <= 0)
        x = min(verts3d[:, 0])
        y = min(verts3d[:, 1])
        x2 = max(verts3d[:, 0])
        y2 = max(verts3d[:, 1])

        return np.array([x, y, x2, y2]), ign

def im_detect_3d_rcnn(im, net, rpn_conf, preprocess, p2, gpu=0, synced=False):
    """
    Object detection in 3D
    """

    imH_orig = im.shape[0]
    imW_orig = im.shape[1]

    im = preprocess(im)

    # move to GPU
    im = torch.from_numpy(im).cuda()

    imH = im.shape[2]
    imW = im.shape[3]

    scale_factor = imH / imH_orig

    cls, prob, bbox_2d, bbox_3d, feat_size, rcnn_info, rcnn_3d = net(im, p2[np.newaxis, :, :], [scale_factor])

    num_classes = len(rpn_conf.lbls) + 1

    if rpn_conf.rcnn_out_mask[0] == 0:
        prob = F.softmax(torch.from_numpy(rcnn_info[:, 13:13 + num_classes]).type(torch.cuda.FloatTensor), dim=1)
    elif rpn_conf.rcnn_out_mask[0] < 1:
        prob = F.softmax(rcnn_3d[:, 0:num_classes] + torch.from_numpy(rcnn_info[:, 13:13+num_classes]).type(torch.cuda.FloatTensor), dim=1)
    else:
        prob = F.softmax(rcnn_3d[:, 0:num_classes], dim=1)

    bbox_x = rcnn_3d[:, num_classes + 0]
    bbox_y = rcnn_3d[:, num_classes + 1]
    bbox_w = rcnn_3d[:, num_classes + 2]
    bbox_h = rcnn_3d[:, num_classes + 3]

    bbox_x3d = rcnn_3d[:, num_classes + 4] * rpn_conf.rcnn_out_mask[5] + torch.from_numpy(rcnn_info[:, 7]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[5])
    bbox_y3d = rcnn_3d[:, num_classes + 5] * rpn_conf.rcnn_out_mask[6] + torch.from_numpy(rcnn_info[:, 8]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[6])
    bbox_z3d = rcnn_3d[:, num_classes + 7] * rpn_conf.rcnn_out_mask[7] + torch.from_numpy(rcnn_info[:, 9]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[7])
    bbox_w3d = rcnn_3d[:, num_classes + 7] * rpn_conf.rcnn_out_mask[8] + torch.from_numpy(rcnn_info[:, 10]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[8])
    bbox_h3d = rcnn_3d[:, num_classes + 8] * rpn_conf.rcnn_out_mask[9] + torch.from_numpy(rcnn_info[:, 11]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[9])
    bbox_l3d = rcnn_3d[:, num_classes + 9] * rpn_conf.rcnn_out_mask[10] + torch.from_numpy(rcnn_info[:, 12]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[10])
    bbox_ry3d = rcnn_3d[:, num_classes + 10] * rpn_conf.rcnn_out_mask[11] + torch.from_numpy(rcnn_info[:, 13]).type(torch.cuda.FloatTensor) * (1 - rpn_conf.rcnn_out_mask[11])

    # detransform 3d
    bbox_x3d = bbox_x3d * rpn_conf.bbox_stds[:, 4][0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * rpn_conf.bbox_stds[:, 5][0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * rpn_conf.bbox_stds[:, 6][0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * rpn_conf.bbox_stds[:, 7][0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * rpn_conf.bbox_stds[:, 8][0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * rpn_conf.bbox_stds[:, 9][0] + rpn_conf.bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * rpn_conf.bbox_stds[:, 10][0] + rpn_conf.bbox_means[:, 10][0]

    # find 3d source
    rois_rcnn = rcnn_info[:, 1:]
    anchors_rcnn = np.hstack((rois_rcnn[:, 0:4], rpn_conf.anchors[rois_rcnn[:, 5].astype(int), 4:]))
    anchors_rcnn[:, 4] = 24.176715662949345
    anchors_rcnn[:, 5] = 1.4076879203200818
    anchors_rcnn[:, 6] = 1.5879433046735336
    anchors_rcnn[:, 7] = 3.2938265088958882
    anchors_rcnn[:, 8] = 0.10889710593269447

    anchors_rcnn = torch.from_numpy(anchors_rcnn).type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = anchors_rcnn[:, 2] - anchors_rcnn[:, 0] + 1.0
    heights = anchors_rcnn[:, 3] - anchors_rcnn[:, 1] + 1.0
    ctr_x = anchors_rcnn[:, 0] + 0.5 * widths
    ctr_y = anchors_rcnn[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d * widths + ctr_x
    bbox_y3d = bbox_y3d * heights + ctr_y
    bbox_z3d = anchors_rcnn[:, 0] + bbox_z3d
    bbox_w3d = torch.exp(bbox_w3d) * anchors_rcnn[:, 1]
    bbox_h3d = torch.exp(bbox_h3d) * anchors_rcnn[:, 2]
    bbox_l3d = torch.exp(bbox_l3d) * anchors_rcnn[:, 3]
    bbox_ry3d = anchors_rcnn[:, 4] + bbox_ry3d

    # bundle
    coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_ry3d), dim=1)

    # compile deltas pred
    deltas_2d = torch.cat((bbox_x[:, np.newaxis], bbox_y[:, np.newaxis], bbox_w[:, np.newaxis], bbox_h[:, np.newaxis]), dim=1)
    coords_2d = bbox_transform_inv(anchors_rcnn, deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

    # detach onto cpu
    coords_2d = coords_2d.cpu().detach().numpy()
    coords_3d = coords_3d.cpu().detach().numpy()
    prob = prob[:, :].cpu().detach().numpy()

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]

    if synced:

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    else:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    return aboxes


def test_kitti_pose(dataset_test, net, conf, results_path, data_path):

    errs_t = []
    errs_r = []

    for seq in conf.vo_test_seqs:

        seq = '{:02d}'.format(seq)
        results_path_seq = results_path
        data_path_seq = os.path.join(data_path, dataset_test, 'sequences', seq, 'image_2', '')

        # make directories
        mkdir_if_missing(results_path_seq, delete_if_exist=True)

        imlist = list_files(data_path_seq, '*.png')
        preprocess = Preprocess([conf.test_scale], conf.image_means, conf.image_stds)

        # init
        test_start = time()

        all_outputs = []

        file = open(os.path.join(results_path_seq, seq + '.txt'), 'w')

        prev_im = None

        for imind, impath in enumerate(imlist):

            im = cv2.imread(impath)
            im = preprocess(im)

            # move to GPU
            im = torch.from_numpy(im).cuda()

            if imind == 0:
                pose_new = np.eye(4, 4)

            else:
                pose = net(torch.cat((im, prev_im), dim=1))
                pose = pose.detach().cpu().numpy()

                pose[0, 0] = (pose[0, 0] + conf.pose_means[0, 0]) * conf.pose_stds[0, 0]
                pose[0, 1] = (pose[0, 1] + conf.pose_means[0, 1]) * conf.pose_stds[0, 1]
                pose[0, 2] = (pose[0, 2] + conf.pose_means[0, 2]) * conf.pose_stds[0, 2]
                pose[0, 3] = (pose[0, 3] + conf.pose_means[0, 3]) * conf.pose_stds[0, 3]
                pose[0, 4] = (pose[0, 4] + conf.pose_means[0, 4]) * conf.pose_stds[0, 4]
                pose[0, 5] = (pose[0, 5] + conf.pose_means[0, 5]) * conf.pose_stds[0, 5]

                pose_rel = euler2mat(pose[0, 3], pose[0, 4], pose[0, 5])
                pose_rel = np.hstack((pose_rel, np.array([[pose[0, 0], pose[0, 1], pose[0, 2]]]).T))
                pose_rel = np.vstack((pose_rel, np.array([[0, 0, 0, 1]])))

                pose_new = inverse_rel_pose(all_outputs[imind - 1], pose_rel)

            all_outputs.append(pose_new)

            prev_im = im

            file.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                pose_new[0, 0], pose_new[0, 1], pose_new[0, 2], pose_new[0, 3],
                pose_new[1, 0], pose_new[1, 1], pose_new[1, 2], pose_new[1, 3],
                pose_new[2, 0], pose_new[2, 1], pose_new[2, 2], pose_new[2, 3],
            ))

            # display stats
            if (imind + 1) % 500 == 0:
                time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
                #logging.info('seq {}, testing {}/{}, dt: {:0.3f}, eta: {}'.format(seq, imind + 1, len(imlist), dt, time_str))

        file.close()

        # evaluate miss rate in Matlab toolbox
        script = os.path.join(data_path, dataset_test, 'devkit', 'cpp', 'evaluate_odometry')
        with open(os.devnull, 'w') as devnull:
            out = subprocess.check_output([script, results_path_seq.replace('/data', ''), seq], stderr=devnull)

        t, r = parse_kitti_vo(os.path.join(results_path_seq.replace('/data', ''), 'stats.txt'))

        errs_t.append(t)
        errs_r.append(r)

        logging.info('seq {}, translation {:.2f}, rotation {:0.8f}'.format(seq, t, r))

    logging.info('translation {:.2f}, rotation {:0.8f}'.format(np.mean(errs_t), np.mean(errs_r)))



def test_kitti_3d(dataset_test, net, rpn_conf, results_path, test_path, has_rcnn=False):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from lib.imdb_util import read_kitti_cal

    imlist = list_files(os.path.join(test_path, dataset_test, 'validation', 'image_2', ''), '*'+rpn_conf.datasets_train[0]['im_ext'])

    preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means, rpn_conf.image_stds)

    # fix paths slightly
    _, test_iter, _ = file_parts(results_path.replace('/data', ''))
    test_iter = test_iter.replace('results_', '')

    debug = False

    if debug:
        mkdir_if_missing('/home/garrick/Desktop/kitti_vis', delete_if_exist=True)

        bev_w = 800
        c1 = (0, 250, 250)
        c2 = (0, 175, 250)
        canvas_bev_orig = create_colorbar(50 * 20, bev_w, color_lo=c1, color_hi=c2)

    # init
    test_start = time()

    for imind, impath in enumerate(imlist):

        im = cv2.imread(impath)

        base_path, name, ext = file_parts(impath)

        # read in calib
        p2 = read_kitti_cal(os.path.join(test_path, dataset_test, 'validation', 'calib', name + '.txt'))
        p2_inv = np.linalg.inv(p2)

        if has_rcnn:
            # forward test batch
            aboxes = im_detect_3d_rcnn(im, net, rpn_conf, preprocess, p2)

        elif ('use_gp' in rpn_conf) and rpn_conf.use_gp:

            # forward test batch
            aboxes, plane = im_detect_3d(im, net, rpn_conf, preprocess, p2)

            plane = plane.detach().cpu().numpy()
            plane *= rpn_conf.plane_stds[0, :].max()
            plane += rpn_conf.plane_means[0, :]

        elif 'depth_map' in rpn_conf and rpn_conf.depth_map is not None:
            depth_map = pickle_read(os.path.join(test_path, dataset_test, 'validation', 'depth_map', name + rpn_conf.depth_map))[:, :, np.newaxis]
            depth_map = torch.from_numpy(depth_map.transpose([2, 0, 1])[np.newaxis, :, :, :]).type(torch.cuda.FloatTensor)
            aboxes = im_detect_3d(im, net, rpn_conf, preprocess, p2, depth_map=depth_map)
        else:
            # forward test batch
            aboxes = im_detect_3d(im, net, rpn_conf, preprocess, p2)

        base_path, name, ext = file_parts(impath)

        file = open(os.path.join(results_path, name + '.txt'), 'w')
        text_to_write = ''
        
        for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):

            box = aboxes[boxind, :]
            score = box[4]
            cls = rpn_conf.lbls[int(box[5] - 1)]

            if ('has_un' in rpn_conf) and rpn_conf.has_un:
                un = box[13]

            if score > rpn_conf.score_thres:

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # for comparison only
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                step_r = 0.3 * math.pi
                r_lim = 0.01

                box_2d = np.array([x1, y1, width, height])

                if rpn_conf.hill_climbing:
                    z3d, ry3d, verts_best = hill_climb(p2, p2_inv, box_2d, x3d, x3d, z3d, w3d, h3d, l3d, ry3d, step_r_init=step_r, r_lim=r_lim)

                while ry3d > math.pi: ry3d -= math.pi * 2
                while ry3d < (-math.pi): ry3d += math.pi * 2

                # predict a more accurate projection
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))

                x3d = coord3d[0]
                y3d = coord3d[1]
                z3d = coord3d[2]

                y3d += h3d/2

                alpha = convertRot2Alpha(ry3d, z3d, x3d)

                if ('has_un' in rpn_conf) and rpn_conf.has_un and ('use_un_for_score' in rpn_conf) and rpn_conf.use_un_for_score:
                    score = un

                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                      + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)
                           
        file.write(text_to_write)
        file.close()

        # display stats
        if (imind + 1) % 1000 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))

    evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf)



def test_kitti_3d_forecast(dataset_test, net, rpn_conf, results_path_pre, results_path_cur, results_path_ego, test_path):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from lib.imdb_util import read_kitti_cal
    from lib.imdb_util import read_kitti_label

    imlist = list_files(os.path.join(test_path, dataset_test, 'validation', 'image_2', ''), '*.png')

    preprocess = Preprocess(rpn_conf.crop_size, rpn_conf.image_means, rpn_conf.image_stds)

    debug = False

    if debug:
        mkdir_if_missing('/home/garrick/Desktop/kitti_vis', delete_if_exist=True)

        bev_w = 800
        c1 = (0, 250, 250)
        c2 = (0, 175, 250)
        canvas_bev_orig = create_colorbar(50 * 20, bev_w, color_lo=c1, color_hi=c2)

    # init
    test_start = time()

    for imind, impath in enumerate(imlist):

        base_path, name, ext = file_parts(impath)

        impath_prev = os.path.join(test_path, dataset_test, 'validation', 'prev_2', name + '_01' + ext)
        impath_prev2 = os.path.join(test_path, dataset_test, 'validation', 'prev_2', name + '_02' + ext)
        impath_prev3 = os.path.join(test_path, dataset_test, 'validation', 'prev_2', name + '_03' + ext)

        im2 = cv2.imread(impath)

        # read in calib
        p2 = read_kitti_cal(os.path.join(test_path, dataset_test, 'validation', 'calib', name + '.txt'))
        p2_inv = np.linalg.inv(p2)

        if debug:
            im_orig = deepcopy(im2)
            im_orig = cv2.resize(im_orig, (im_orig.shape[1] * 2, im_orig.shape[0] * 2))

            canvas_bev = deepcopy(canvas_bev_orig)

            gts = read_kitti_label(os.path.join(test_path, dataset_test, 'validation', 'label_2', name + '.txt'), p2, use_3d_for_2d=True)
            igns, rmvs = determine_ignores(gts, rpn_conf.lbls, rpn_conf.ilbls, rpn_conf.min_gt_vis, rpn_conf.min_gt_h, rpn_conf.max_gt_h)
            gts_full = bbXYWH2Coords(np.array([gt['bbox_full'] for gt in gts]))
            gts_3d = np.array([gt['bbox_3d'] for gt in gts])

            has_gt = gts_full.shape[0] > 0

            if has_gt:
                gts_full = gts_full[(rmvs == False), :]
                gts_3d = gts_3d[(rmvs == False), :]

        h_before = im2.shape[0]
        im2 = preprocess(im2)
        im2 = torch.from_numpy(im2).cuda()

        if rpn_conf.video_det:

            if rpn_conf.video_count >= 2:
                im1 = cv2.imread(impath_prev)
                im1 = preprocess(im1)
                im1 = torch.from_numpy(im1).cuda()
                im2 = torch.cat((im2, im1), dim=0)

            if rpn_conf.video_count >= 3:
                im0 = cv2.imread(impath_prev2)
                if im0 is None: im0 = cv2.imread(impath_prev)
                im0 = preprocess(im0)
                im0 = torch.from_numpy(im0).cuda()
                im2 = torch.cat((im2, im0), dim=0)

            if rpn_conf.video_count >= 4:
                im0 = cv2.imread(impath_prev3)
                if im0 is None: im0 = cv2.imread(impath_prev)
                im0 = preprocess(im0)
                im0 = torch.from_numpy(im0).cuda()
                im2 = torch.cat((im2, im0), dim=0)

        scale_factor = im2.shape[1] / h_before

        # forward test batch
        aboxes_all = im_detect_3d_forecast(im2, net, rpn_conf, scale_factor, p2)

        base_path, name, ext = file_parts(impath)

        for bind in range(len(aboxes_all)):

            if bind==0:
                results_path = results_path_cur
            elif bind==1:
                results_path = results_path_ego
            else:
                results_path = results_path_pre

            aboxes = aboxes_all[bind]

            # first write normal_predictions
            file = open(os.path.join(results_path, name + '.txt'), 'w')
            text_to_write = ''

            for boxind in range(0, min(rpn_conf.nms_topN_post, len(aboxes))):

                box = aboxes[boxind, :]
                score = box[4]
                cls = rpn_conf.lbls[int(box[5] - 1)]

                if ('has_un' in rpn_conf) and rpn_conf.has_un:
                    un = box[13]

                if score > rpn_conf.score_thres:

                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    width = (x2 - x1 + 1)
                    height = (y2 - y1 + 1)

                    if debug and has_gt:
                        ols = iou(np.array([[x1, y1, x2, y2]]), gts_full)[0, :]
                        gtind = np.argmax(ols)
                        ol = np.amax(ols)

                    # for comparison only
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # plot 3D box
                    x3d = box[6]
                    y3d = box[7]
                    z3d = box[8]
                    w3d = box[9]
                    h3d = box[10]
                    l3d = box[11]
                    ry3d = box[12]

                    if not (('is_kalman' in rpn_conf) and rpn_conf.is_kalman and bind == 1):
                        # convert alpha into ry3d
                        coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                        ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                        x3d = coord3d[0]
                        y3d = coord3d[1]
                        z3d = coord3d[2]

                    success = False

                    # match
                    if debug and has_gt and ol > 0.25:
                        gt_x2d = gts_3d[gtind, 0]
                        gt_y2d = gts_3d[gtind, 1]
                        gt_z2d = gts_3d[gtind, 2]
                        gt_w3d = gts_3d[gtind, 3]
                        gt_h3d = gts_3d[gtind, 4]
                        gt_l3d = gts_3d[gtind, 5]
                        gt_x3d = gts_3d[gtind, 7]
                        gt_y3d = gts_3d[gtind, 8]
                        gt_z3d = gts_3d[gtind, 9]
                        gt_rotY = gts_3d[gtind, 10]

                        gt_x3d_bot = gt_x3d
                        gt_y3d_bot = gt_y3d + gt_h3d / 2
                        gt_z3d_bot = gt_z3d

                        gt_x2d_bot, gt_y2d_bot, gt_z2d_bot = project_3d_point(p2, gt_x3d_bot, gt_y3d_bot, gt_z3d_bot)

                        gt_x1 = gts_full[gtind, 0]
                        gt_y1 = gts_full[gtind, 1]
                        gt_x2 = gts_full[gtind, 2]
                        gt_y2 = gts_full[gtind, 3]
                        gt_w = gt_x2 - gt_x1 + 1
                        gt_h = gt_y2 - gt_y1 + 1
                        box_gt = np.array([gt_x1, gt_y1, gt_w, gt_h])

                        verts_cur, corners_3d_cur = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)
                        verts_gts, corners_3d_gt = project_3d(p2, gt_x3d, gt_y3d, gt_z3d, gt_w3d, gt_h3d, gt_l3d, gt_rotY, return_3d=True)

                        iou_bev, iou_3d = iou3d(corners_3d_cur, corners_3d_gt, gt_h3d * gt_l3d * gt_w3d + h3d * l3d * w3d)

                        cen_err = np.sqrt((z3d - gt_z3d) ** 2 + (gt_x3d - x3d) ** 2 + (gt_y3d - y3d) ** 2)

                        if cls == 'Pedestrian' or cls == 'Cyclist': success = iou_3d >= 0.5
                        elif cls == 'Car': success = iou_3d >= 0.7
                        else: print('Unknown class ' + cls)


                    if bind==0 and debug:

                        if int(box[5] - 1) == 0: c = (0, 242, 242); short='C'
                        if int(box[5] - 1) == 2: c = (0, 168, 255); short='B'
                        if int(box[5] - 1) == 1: c = (150, 255, 40); short='P'

                        c = (240, 80, 80) if success else (0, 0, 240)

                        verts3d, corners_3d = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)
                        invalid = np.any(corners_3d[2, :] <= 0)

                        bev_scale = 20

                        if not invalid:
                            draw_3d_box(im_orig, 2*verts3d, color=c, thickness=2)
                            draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=c, scale=bev_scale, thickness=3)

                        x_tmp = min(2 * verts3d[:, 0])
                        y_tmp = min(2 * verts3d[:, 1])

                        if has_gt and ol>0.25:
                            draw_bev(canvas_bev, gt_z3d, gt_l3d, gt_w3d, gt_x3d, gt_rotY, color=(0, 180, 0), scale=bev_scale, thickness=2)
                            text = '{} {:.2f}, {:.2f}'.format(short, iou_3d, cen_err)
                        else:
                            text = '{}'.format(short)

                        draw_text(im_orig, text, [x_tmp, y_tmp], bg_color=c, scale=0.75, lineType=2, blend=0.25)

                        #draw_2d_box(im_orig, 2*np.array([x1, y1, width, height]), c, thickness=2)

                        #x_tmp = min(2 * verts3d[:, 0])
                        #y_tmp = min(2 * verts3d[:, 1])
                        #draw_text(im, '{:.2f}, {:.2f}'.format(iou_3d, cen_err), [x_tmp, y_tmp], bg_color=c, scale=0.75, lineType=2, blend=0.25)

                    y3d += h3d / 2

                    alpha = convertRot2Alpha(ry3d, z3d, x3d)

                    if ('has_un' in rpn_conf) and rpn_conf.has_un and ('use_un_for_score' in rpn_conf) and rpn_conf.use_un_for_score:
                        score = un

                    text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                      + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

            if debug:
                canvas_bev = cv2.flip(canvas_bev, 0)

                # draw tick marks
                draw_tick_marks(canvas_bev, [50, 40, 30, 20, 10, 0])

                # information for left corner
                draw_text(im_orig, '(3D IoU, cen_dist), r=miss, b=match, g=groundtruth', [0, 25], scale=1, lineType=2)

                im_orig = imhstack(im_orig, canvas_bev)
                imwrite(im_orig, '/home/garrick/Desktop/kitti_vis/{:06d}.jpg'.format(imind))

            file.write(text_to_write)
            file.close()

        # display stats
        if (imind + 1) % 500 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))


    for bind in range(len(aboxes_all)):

        if bind == 0:
            results_path = results_path_cur
        elif bind == 1:
            results_path = results_path_ego
        else:
            results_path = results_path_pre

        _, test_iter, _ = file_parts(results_path.replace('/data', ''))
        test_iter = test_iter.replace('results_', '')

        evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf)


def parse_kitti_result(respath, use_40=False):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    if use_40:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])
    else:
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])

    return easy, mod, hard


def parse_kitti_vo(respath):

    text_file = open(respath, 'r')

    acc = np.zeros([1, 2], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    t = acc[0, 0]*100
    r = acc[0, 1]

    return t, r


def test_kitti_2d(dataset_test, net, rpn_conf, results_path, test_path):
    """
    Test the KITTI framework for object detection in 3D
    """

    imlist = list_files(os.path.join(test_path, dataset_test, 'validation', 'image_2', ''), '*.png')

    preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means, rpn_conf.image_stds)

    # fix paths slightly
    _, test_iter, _ = file_parts(results_path.replace('/data', ''))
    test_iter = test_iter.replace('results_', '')

    # init
    test_start = time()

    for imind, impath in enumerate(imlist):

        im = cv2.imread(impath)

        base_path, name, ext = file_parts(impath)

        # forward test batch
        aboxes = im_detect_2d(im, net, rpn_conf, preprocess)

        base_path, name, ext = file_parts(impath)

        file = open(os.path.join(results_path, name + '.txt'), 'w')

        for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):

            box = aboxes[boxind, :]
            score = box[4]
            cls = rpn_conf.lbls[int(box[5] - 1)]

            if score > 0.75:

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                w = (x2 - x1 + 1)
                h = (y2 - y1 + 1)

                file.write(('{} -1 -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                           + '{:.6f} {:.6f}\n').format(cls, x1, y1, x2, y2, -1, -1, -1, -1, -1, -1, -1, score))

        file.close()

        # display stats
        if (imind + 1) % 1000 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))


    # evaluate
    script = os.path.join(test_path, dataset_test, 'devkit', 'cpp', 'evaluate_object')
    with open(os.devnull, 'w') as devnull:
        out = subprocess.check_output([script, results_path.replace('/data', '')], stderr=devnull)

    if os.path.exists(os.path.join(results_path.replace('/data', ''), 'stats_car_detection.txt')):

        text_file = open(os.path.join(results_path.replace('/data', ''), 'stats_car_detection.txt'), 'r')

        acc = np.zeros([3, 41], dtype=float)

        lind = 0
        for line in text_file:

            parsed = re.findall('([\d]+\.?[\d]*)', line)

            for i, num in enumerate(parsed):
                acc[lind, i] = float(num)

            lind += 1

        text_file.close()

        easy2d = np.mean(acc[0, 0:41:4])
        mod2d = np.mean(acc[1, 0:41:4])
        hard2d = np.mean(acc[2, 0:41:4])

        logging.info('test_iter {} 2d car --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, easy2d,
                                                                                                 mod2d, hard2d))

    if os.path.exists(os.path.join(results_path.replace('/data', ''), 'stats_pedestrian_detection.txt')):

        text_file = open(os.path.join(results_path.replace('/data', ''), 'stats_pedestrian_detection.txt'), 'r')

        acc = np.zeros([3, 41], dtype=float)

        lind = 0
        for line in text_file:

            parsed = re.findall('([\d]+\.?[\d]*)', line)

            for i, num in enumerate(parsed):
                acc[lind, i] = float(num)

            lind += 1

        text_file.close()

        easy2d = np.mean(acc[0, 0:41:4])
        mod2d = np.mean(acc[1, 0:41:4])
        hard2d = np.mean(acc[2, 0:41:4])

        logging.info('test_iter {} 2d ped --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, easy2d,
                                                                                                 mod2d, hard2d))

    if os.path.exists(os.path.join(results_path.replace('/data', ''), 'stats_cyclist_detection.txt')):

        text_file = open(os.path.join(results_path.replace('/data', ''), 'stats_cyclist_detection.txt'), 'r')

        acc = np.zeros([3, 41], dtype=float)

        lind = 0
        for line in text_file:

            parsed = re.findall('([\d]+\.?[\d]*)', line)

            for i, num in enumerate(parsed):
                acc[lind, i] = float(num)

            lind += 1

        text_file.close()

        easy2d = np.mean(acc[0, 0:41:4])
        mod2d = np.mean(acc[1, 0:41:4])
        hard2d = np.mean(acc[2, 0:41:4])

        logging.info('test_iter {} 2d cyc --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'.format(test_iter, easy2d,
                                                                                                 mod2d, hard2d))


def run_kitti_eval_script(script_path, results_data, lbls, use_40=True):

    # evaluate primary experiment
    with open(os.devnull, 'w') as devnull:
        _ = subprocess.check_output([script_path, results_data], stderr=devnull)

    results_obj = edict()

    for lbl in lbls:

        lbl = lbl.lower()

        respath_2d = os.path.join(results_data, 'stats_{}_detection.txt'.format(lbl))
        respath_or = os.path.join(results_data, 'stats_{}_orientation.txt'.format(lbl))
        respath_gr = os.path.join(results_data, 'stats_{}_detection_ground.txt'.format(lbl))
        respath_3d = os.path.join(results_data, 'stats_{}_detection_3d.txt'.format(lbl))

        if os.path.exists(respath_2d):
            easy, mod, hard = parse_kitti_result(respath_2d, use_40=use_40)
            results_obj['det_2d_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_or):
            easy, mod, hard = parse_kitti_result(respath_or, use_40=use_40)
            results_obj['or_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_gr):
            easy, mod, hard = parse_kitti_result(respath_gr, use_40=use_40)
            results_obj['gr_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_3d):
            easy, mod, hard = parse_kitti_result(respath_3d, use_40=use_40)
            results_obj['det_3d_' + lbl] = [easy, mod, hard]

    return results_obj


def evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf, use_logging=True):

    # evaluate primary experiment
    script = os.path.join(test_path, dataset_test, 'devkit', 'cpp', 'evaluate_object')

    results_obj = edict()

    task_keys = ['det_2d', 'or', 'gr', 'det_3d']

    # main experiment results
    results_obj.main = run_kitti_eval_script(script, results_path.replace('/data', ''), rpn_conf.lbls, use_40=1)

    # print main experimental results for each class, and each task
    for lbl in rpn_conf.lbls:

        lbl = lbl.lower()

        for task in task_keys:

            task_lbl = task + '_' + lbl

            if task_lbl in results_obj.main:

                easy = results_obj.main[task_lbl][0]
                mod = results_obj.main[task_lbl][1]
                hard = results_obj.main[task_lbl][2]

                print_str = 'test_iter {} {} {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'\
                    .format(test_iter, lbl, task.replace('det_', ''), easy, mod, hard)

                if use_logging:
                    logging.info(print_str)
                else:
                    print(print_str)

    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    dis_keys = ['15', '30', '45', '60']

    print('working on distance=', end='')

    for dis_key in dis_keys:

        print(dis_key, end='', flush=True)

        for iou_key in iou_keys:

            eval_key = 'evaluate_object_{}m_{}'.format(dis_key, iou_key)
            save_key = 'res_{}m_{}'.format(dis_key, iou_key)

            print('.', end='', flush=True)

            script = os.path.join(test_path, dataset_test, 'devkit', 'cpp', eval_key)
            tmp_obj = run_kitti_eval_script(script, results_path.replace('/data', ''), rpn_conf.lbls)
            results_obj[save_key] = tmp_obj

    print('')

    pickle_write(results_path.replace('/data', '/') + 'results_obj', results_obj)

    try:
        backend = plt.get_backend()
        plt.switch_backend('agg')
        save_kitti_ROC(results_obj, results_path.replace('/data', '/'), rpn_conf.lbls)
        print_kitti_ROC(results_obj, test_iter, rpn_conf.lbls, use_logging=use_logging)
        plt.switch_backend(backend)
    except:
        pass


def print_kitti_ROC(results_obj, test_iter, lbls, use_logging=True):


    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    iou_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dis_keys = ['15', '30', '45', '60']

    for lbl in lbls:

        lbl = lbl.lower()

        for dis in dis_keys:

            roc = [results_obj['res_{}m_{}'.format(dis, iou)]['det_3d_' + lbl][1] for iou in iou_keys]
            val = np.array(roc).mean()
            if lbl == 'car': legend = '{}m (av={:.4f}, 0.7={:.4f})'.format(dis, val, roc[6])
            else: legend = '{}m (av={:.4f}, 0.5={:.4f})'.format(dis, val, roc[4])

            if use_logging:
                logging.info(test_iter + ' ' + lbl + ' ' + legend)
            else:
                print(test_iter + ' ' + lbl + ' ' + legend)


def save_kitti_ROC(results_obj, folder, lbls):

    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    iou_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dis_keys = ['15', '30', '45', '60']

    for lbl in lbls:

        lbl = lbl.lower()

        for dis in dis_keys:

            roc = [results_obj['res_{}m_{}'.format(dis, iou)]['det_3d_' + lbl][1] for iou in iou_keys]
            val = np.array(roc).mean()
            if lbl == 'car': legend = '{}m (av={:.4f}, 0.7={:.4f})'.format(dis, val, roc[6])
            else: legend = '{}m (av={:.4f}, 0.5={:.4f})'.format(dis, val, roc[4])

            plt.plot(iou_vals, roc, label=legend)

        plt.xlabel('3D IoU Criteria')
        plt.ylabel('AP')
        plt.legend()

        roc_path = os.path.join(folder, 'roc_' + lbl + '.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.clf()


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    #cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    ol = iou(b1, b2)[0][0]
    #ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) + np.abs(y2 - y2_new))

    return ol, verts3d, b2, invalid


def test_caltech(net, rpn_conf, results_path, data_path, test_db, gpu=0):
    """
    Tests the caltech framework for pedestrian detection.
    """

    imlist = list_files(os.path.join(data_path, 'test', 'images', ''), '*.jpg')

    preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means, rpn_conf.image_stds)

    # init
    last_set = ''
    test_start = time()

    debug = False

    for imind, impath in enumerate(imlist):

        im = cv2.imread(impath)

        # forward test batch
        aboxes = im_detect_2d(im, net, rpn_conf, preprocess, gpu=gpu)

        # parse caltech set_V000_I00000 name
        base_path, name, ext = file_parts(impath)
        caltech_pat = re.compile('(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)')

        parsed = caltech_pat.fullmatch(name)
        setname = parsed.group(1)
        vname = parsed.group(2)
        iname = parsed.group(3)
        inum = int(iname) + 1

        curr_set = setname + vname

        mkdir_if_missing(os.path.join(results_path, setname))

        # close file
        if last_set != '' and curr_set != last_set:
            file.close()

        # need to open new file
        if curr_set != last_set:
            file = open(os.path.join(results_path, setname, vname + '.txt'), 'w')
            last_set = curr_set

        # write detection boxes
        for bind in range(0, min(aboxes.shape[0], rpn_conf.nms_topN_post)):

            x1 = aboxes[bind, 0]
            y1 = aboxes[bind, 1]
            x2 = aboxes[bind, 2]
            y2 = aboxes[bind, 3]

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            score = aboxes[bind, 4]

            if score > 0.001 and h >= 40:

                file.write('{:d},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(inum, x1, y1, w, h, score))

                if debug and score > 0.95:

                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (50, 255, 0), 2)
                    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    pass


        # display stats
        if (imind + 1) % 1000 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))

    file.close()

    # evaluate miss rate in Matlab toolbox
    base_path = os.getcwd()
    os.chdir("external/caltech_toolbox")
    script = 'evaluate_result_dir_all {} {}'.format(results_path, test_db)
    out = subprocess.check_output(['matlab', '-r', script, '-nojvm', '-nodesktop', '-nosplash'])

    # return to base dir
    os.chdir(base_path)

    # parse and print miss rate
    accs = []

    accs.append({'name': 'Reasonable',
                'mr': float(re.search('1, scores=([-+]?\d*\.\d+|\d+)', str(out)).group(1)),
                'recall': float(re.search('1, recall=([-+]?\d*\.\d+|\d+)', str(out)).group(1))
                })

    accs.append({'name': 'Scale=medium',
                'mr': float(re.search('5, scores=([-+]?\d*\.\d+|\d+)', str(out)).group(1)),
                'recall': float(re.search('5, recall=([-+]?\d*\.\d+|\d+)', str(out)).group(1))
                })

    accs.append({'name': 'Occ=Partial',
                'mr': float(re.search('8, scores=([-+]?\d*\.\d+|\d+)', str(out)).group(1)),
                'recall': float(re.search('8, recall=([-+]?\d*\.\d+|\d+)', str(out)).group(1))
                })

    return accs

