"""
This file is meant to contain 3D geometry math functions
"""

import os
import sys
from glob import glob
from time import time
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle
import logging
import datetime
import pprint
import shutil
import math
import torch
import copy
import cv2
#from scipy.spatial.transform import Rotation as scipy_R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import random
from itertools import combinations


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def project_3d_point(p2, x3d, y3d, z3d):

    coord2d = p2.dot(np.array([x3d, y3d, z3d, 1]))
    coord2d[:2] /= coord2d[2]

    return coord2d[0], coord2d[1], coord2d[2]


def backproject_3d_point(p2_inv, x2d, y2d, z2d):

    coord3d = p2_inv.dot(np.array([x2d * z2d, y2d * z2d, z2d, 1]))

    return coord3d[0], coord3d[1], coord3d[2]


def ray_trace(p2_inv, x, y, plane):

    # find 2 points to define ray
    point1 = p2_inv[0:3, 3]
    if np.array(x).size > 1:
        point2 = p2_inv.dot(np.pad(np.array([x, y]), [(0, 2), (0, 0)], mode='constant', constant_values=1))[0:3, :]
    else:
        point2 = p2_inv.dot(np.array([x, y, 1, 1]))[0:3]

    # compute a point on the plane_gt where x=0, z=0
    # ax + by + cz = d
    # y = (d - ax - cz)/b
    plane_point = np.zeros(3, dtype=float)
    plane_point[1] = (plane[3]) / plane[1]

    # compute point line intersects with plane_gt
    pt_intersect = compute_plane_line_intersection(plane[:3], plane_point, point1, point2)

    return pt_intersect

def ray_trace(p2_inv, x, y, plane):

    # find 2 points to define ray
    point1 = p2_inv[0:3, 3]
    if np.array(x).size > 1:
        point2 = p2_inv.dot(np.pad(np.array([x, y]), [(0, 2), (0, 0)], mode='constant', constant_values=1))[0:3, :]
    else:
        point2 = p2_inv.dot(np.array([x, y, 1, 1]))[0:3]

    # compute a point on the plane_gt where x=0, z=0
    # ax + by + cz = d
    # y = (d - ax - cz)/b
    if len(plane.shape) > 1:
        plane_point = np.zeros([3, plane.shape[0]], dtype=float)
    else:
        plane_point = np.zeros(3, dtype=float)

    plane_point[1] = (plane.T[3]) / plane.T[1]

    # compute point line intersects with plane_gt
    pt_intersect = compute_plane_line_intersection(plane.T[:3], plane_point, point1, point2)

    return pt_intersect


def equation_plane(point1, point2, point3):
    a1 = point2[0] - point1[0]
    b1 = point2[1] - point1[1]
    c1 = point2[2] - point1[2]
    a2 = point3[0] - point1[0]
    b2 = point3[1] - point1[1]
    c2 = point3[2] - point1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2

    # ax + by + cz = d
    d = (a * point1[0] + b * point1[1] + c * point1[2])

    return (a, b, c, d)


def equation_plane2(p1, p2, p3):

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # ax + by + cz = ds
    d = np.dot(cp, p3)

    vec = (a, b, c, d)

    return vec


def points_dist(p1, p2):

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)

    return dist


# Function to find distance
def plane_distance(points, plane):

    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = -plane[3]

    x1 = points.T[0]
    y1 = points.T[1]
    z1 = points.T[2]

    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))

    return d/e


def compute_plane_rotation(plane1, plane2):

    vec_cro = np.cross(plane1[0:3], plane2[0:3])
    vec_dot = np.dot(plane1[0:3], plane2[0:3])

    x = vec_cro[0]  # (v1 x v2).x
    y = vec_cro[1]  # (v1 x v2).y
    z = vec_cro[2]  # (v1 x v2).z

    w = norm(plane1[0:3]) * norm(plane2[0:3]) + vec_dot  # | v1 | | v2 | + v1•v2

    r = scipy_R.from_quat([x, y, z, w])

    return r

def solve_transform(X, Y, compute_error=False):

    R, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    if compute_error:
        err = np.abs(Y - X.dot(R))
        return R.T, err
    else:
        return R.T


def planeFit(points):
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T)
    return ctr, np.linalg.svd(M)[0][:,-1]


def plane_fit(points):

    if points.shape[0] == 1:
        plane = np.zeros([4])
        plane[1] = 1
        plane[-1] = points[0, 1]
        return plane

    elif points.shape[0] == 2:
        #A = points[:, 0:3:2]
        #B = 1 - points[:, 1]
        #n, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        #n = np.insert(n, 1, 1)
        A = points[:, 0:2]
        n, _, _, _ = np.linalg.lstsq(A, np.ones(points.shape[0]), rcond=None)
        n = normalize(np.insert(n, 2, 0))

    else:
        # compute normal
        n, _, _, _ = np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)
        n = normalize(n)

    # compute plane
    center = points.mean(axis=0)
    plane = np.hstack((n, n.dot(center)))

    return plane


def plane_eval(p2_inv, plane, points2d, points3d):

    #errors = plane_distance(points3d, plane)

    point_in = ray_trace(p2_inv, points2d.T[0], points2d.T[1], plane)
    errors = np.sqrt(((point_in - points3d.T)**2).sum(axis=0))
    #num_gts += 1
    #if not np.isnan(point_in).any():
    #    err_all = np.abs(point_in - gt.bottom_3d)

    return errors


def ransac(p2_inv, points2d, points3d, max_iters=1000, min_n=3, t_inlier=0.5, verbose=False, percent_required=0.50, only_inliers=False):

    n_points = points3d.shape[0]
    n_required = (n_points - min_n)*percent_required

    # init
    best_plane = plane_fit(points3d)
    best_err = plane_eval(p2_inv, best_plane, points2d, points3d).mean()

    # not enough points?
    if n_points <= min_n: return best_plane

    combs = set(combinations(range(n_points), min_n))

    # too many?
    if max_iters < len(combs):
        combs = random.sample(combs, k=min(max_iters, len(combs)))

    for iter, sel_inds in enumerate(combs):

        points_inds = list(sel_inds)

        # fit model subset
        plane_sub = plane_fit(points3d[points_inds])

        # compute error on all points
        errors = plane_eval(p2_inv, plane_sub, points2d, points3d)
        error = errors.mean()

        # check for good fit
        if not only_inliers and error < best_err:
            if verbose: print('Improved at iter {}. error {:.4f} -> {:.4f}'.format(iter, best_err, error))
            best_plane = plane_sub
            best_err = error

        # find inliers
        inliers = np.array([i for i in range(n_points) if errors[i] < t_inlier and not (i in points_inds)])

        # decent fit?
        if inliers.shape[0] >= n_required:

            # select
            points_more = np.hstack((points_inds, inliers))

            # fit
            plane_more = plane_fit(points3d[points_more])

            # optimize model on inliers and eval only on inliers
            if only_inliers:
                errors_more = plane_eval(p2_inv, plane_more, points2d[points_more], points3d[points_more])

            # otherwise, optimize model on inliers and eval on all
            else:
                errors_more = plane_eval(p2_inv, plane_more, points2d, points3d)

            error_more = errors_more.mean()

            # check for good fit
            if error_more < best_err:
                if verbose: print('Improved at iter {}. error {:.4f} -> {:.4f}'.format(iter, best_err, error_more))
                best_plane = plane_more
                best_err = error_more

    return best_plane


def ransac_clusters(p2_inv, points2d, points3d, max_iters=1000, min_n=3, t_match=0.05, t_inlier=0.25, verbose=False, percent_required=0.50, only_inliers=False):

    n_points = points3d.shape[0]
    n_required = (n_points - min_n)*percent_required

    if n_points == 1:
        plane = np.zeros([1, 4])
        plane[0, 1] = 1
        plane[0,-1] = points3d[0, 1]
        return plane

    elif n_points == 2:
        A = points3d[:, 0:2]
        n, _, _, _ = np.linalg.lstsq(A, np.ones(n_points), rcond=None)
        n = normalize(np.insert(n, 2, 0))

        # compute plane
        center = points3d.mean(axis=0)
        plane = np.hstack((n, n.dot(center)))
        return np.tile(plane[np.newaxis, :], [n_points, 1])

    # init
    best_plane = plane_fit(points3d)
    errors = plane_eval(p2_inv, best_plane, points2d, points3d)
    best_err = errors.mean()

    best_planes = np.tile(best_plane[np.newaxis, :], [n_points, 1])
    best_errs = np.inf*np.ones(n_points)

    # check individual gts
    for i in range(n_points):

        if errors[i] < t_match and best_err < best_errs[i]:
            best_errs[i] = best_err

    # not enough points?
    if n_points <= min_n: return best_planes

    combs = set(combinations(range(n_points), min_n))

    # too many?
    if max_iters < len(combs):
        combs = random.sample(combs, k=min(max_iters, len(combs)))

    for iter, sel_inds in enumerate(combs):

        points_inds = list(sel_inds)

        # fit model subset
        plane_sub = plane_fit(points3d[points_inds])

        # compute error on all points
        errors = plane_eval(p2_inv, plane_sub, points2d, points3d)
        error = errors.mean()

        # check for good fit
        if not only_inliers and error < best_err:
            if verbose: print('Improved at iter {}. error {:.4f} -> {:.4f}'.format(iter, best_err, error))
            best_plane = plane_sub
            best_err = error

        # check individual gts
        for i in range(n_points):

            if errors[i] < t_match and error < best_errs[i]:
                best_planes[i] = plane_sub
                best_errs[i] = error

        # find inliers
        inliers = np.array([i for i in range(n_points) if errors[i] < t_inlier and not (i in points_inds)])

        # decent fit?
        if inliers.shape[0] >= n_required:

            # select
            points_more = np.hstack((points_inds, inliers))

            # fit
            plane_more = plane_fit(points3d[points_more])

            # optimize model on inliers and eval only on inliers
            if only_inliers:
                errors_more = plane_eval(p2_inv, plane_more, points2d[points_more], points3d[points_more])
                error_more = errors_more.mean()

            # otherwise, optimize model on inliers and eval on all
            else:
                errors_more = plane_eval(p2_inv, plane_more, points2d, points3d)
                error_more = errors_more.mean()

            # check for good fit
            if error_more < best_err:
                if verbose: print('Improved at iter {}. error {:.4f} -> {:.4f}'.format(iter, best_err, error_more))
                best_plane = plane_more
                best_err = error_more

            # check individual gts
            for errind in range(errors_more.shape[0]):

                i = errind if not only_inliers else points_more[errind]

                if errors_more[errind] < t_match and error_more < best_errs[i]:
                    best_planes[i] = plane_more
                    best_errs[i] = error_more

    return best_planes


def fit_plane2(voxels, iterations=50, inlier_thresh=10):  # voxels : x,y,z
    inliers, planes = [], []
    xy1 = np.concatenate([voxels[:, :-1], np.ones((voxels.shape[0], 1))], axis=1)
    z = voxels[:, -1].reshape(-1, 1)
    for _ in range(iterations):
        random_pts = voxels[np.random.choice(voxels.shape[0], min(voxels.shape[1]*10, voxels.shape[0]), replace=False), :]
        plane_transformation, residual = fit_pts_to_plane(random_pts)
        inliers.append(((z - np.matmul(xy1, plane_transformation)) <= inlier_thresh).sum())
        planes.append(plane_transformation)
    return planes[np.array(inliers).argmax()]


def fit_pts_to_plane(voxels):  # x y z  (m x 3)
    # https: // math.stackexchange.com / questions / 99299 / best - fitting - plane - given - a - set - of - points
    xy1 = np.concatenate([voxels[:, :-1], np.ones((voxels.shape[0], 1))], axis=1)
    z = voxels[:, -1].reshape(-1, 1)
    fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
    errors = z - np.matmul(xy1, fit)
    residual = np.linalg.norm(errors)
    return fit, residual


def projection_ray_trace(p2, p2_inv, x2d, y2d, y3d):

    intrin_d = p2[1, 1]
    intrin_e = p2[1, 2]
    intrin_f = p2[1, 3]
    intrin_h = p2[2, 3]

    z2d = (intrin_d * y3d - intrin_e * intrin_h + intrin_f) / (y2d - intrin_e)

    #z2d = z2d.clip(5, 30)

    if np.array(x2d).size == 0:
        coord3d = p2_inv.dot(np.array([x2d * z2d, y2d * z2d, 1 * z2d, 1]))
    else:
        coord3d = p2_inv.dot(np.vstack((x2d * z2d, y2d * z2d, z2d, np.ones(z2d.shape))))

    return z2d, coord3d


def compute_plane_line_intersection(n, V0, P0, P1):
    # n: normal vector of the Plane
    # V0: any point that belongs to the Plane
    # P0: end point 1 of the segment P0P1
    # P1:  end point 2 of the segment P0P1
    #n = np.array([1., 1., 1.])
    #V0 = np.array([1., 1., -5.])
    #P0 = np.array([-5., 1., -1.])
    #P1 = np.array([1., 2., 3.])

    if len(n.shape) > 1:
        w = P0[:, np.newaxis] - V0
        u = P1.T - P0
        N = -(n * w).sum(axis=0)
        D = (n * u.T).sum(axis=0)

    else:
        w = P0 - V0
        u = P1.T - P0
        N = -np.dot(n, w)
        D = np.dot(n, u.T)

    sI = N / D
    I = (P0 + (sI * u.T).T).T

    return I

def norm(vec):
    return np.linalg.norm(vec)


def normalize(vec):

    vec /= norm(vec)
    return vec

def snap_to_pi(ry3d):

    if type(ry3d) == torch.Tensor:
        while (ry3d > (math.pi)).any(): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while (ry3d <= (-math.pi)).any(): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    else:

        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d <= (-math.pi): ry3d += math.pi * 2

    return ry3d

