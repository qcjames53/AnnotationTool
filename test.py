import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)


def imread(path):

    return cv2.imread(path)


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x4
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

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):

    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)


def imshow(im, fig_num=None):

    if fig_num is not None: plt.figure(fig_num)

    if len(im.shape) == 2:
        im = np.tile(im, [3, 1, 1]).transpose([1, 2, 0])

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show(block=False)


def main():
    im = imread("test_image.png")

    # P = np.array([[-256.225938736042, -2846.91854827588, -0.275163615602025, 0],
    # [2985.16394007471, -94.2406823999176, -0.0931442548133298, 0],
    # [614.450521678984, -586.119436560467, 0.956874669141205, 0],
    # [888783.157118041, 983556.246007256, 10011.8623890279, 1]])
    # T = np.array([[0.0124866622028966,	-0.922724814060489,	0.385257058058730, -949.716834496361],
    #             [0.968318255488066,	0.107259244918172,	0.225510998552470, -289.973009173372],
    #             [-0.249406975354699,	0.370235562711931,	0.894830592206033, 10607.2908443277]])

    # T = np.array([[-0.208408426360754,	0.238719828072300,	0.948461265158976, -984.855042850143],
    #             [0.976619596416243,	-0.00148562062215848,	0.214969664899822, -89.1277322231390],
    #             [0.0527265750604841,	0.971087347573069,	-0.232828846292492, 9185.11588014885]])

    # T = np.array([[-0.209068665581356,	0.238664694728465,	0.948329824777329, -984.450367952150],
    #             [-0.0530269139060926,	-0.971100658068604,	0.232705088685918, -89.0851246291459],
    #             [0.976462205860333,	-0.00163566160749117,	0.215682370947370, 9181.58714604201]])

    # T = np.array([[-0.209068665581356,-0.0530269139060926,0.976462205860333,-984.450367952150],
    #                 [0.238664694728465,-0.971100658068604,-0.00163566160749117, -89.0851246291459],
    #                 [0.948329824777329,0.232705088685918,0.215682370947370, 9181.58714604201]])

    T = np.array([[0.948329824777329,0.232705088685918,0.215682370947370, -984.450367952150],
    [0.238664694728465,-0.971100658068604,-0.00163566160749117, -89.0851246291459],
    [-0.209068665581356,-0.0530269139060926,0.976462205860333, 9181.58714604201]])

    K = np.array([[3093.39630281130,	0,	0],
                [0,	2955.67509422577,	0],
                [381.301968789228,	169.658566416672,	1]])

    P = np.matmul(K.transpose(), T)


    x3d = 100.0
    y3d = 400.0
    z3d = 1000.0

    w3d = 100.0
    h3d = 100.0
    l3d = 100.0
    rotY = 0
    plt.close('all')
    verts3d = project_3d(P, x3d, y3d, z3d, w3d, h3d, l3d, rotY)
    draw_3d_box(im, verts3d, color=(255, 0, 0))
    imshow(im)

    a =1


if __name__ == '__main__':
    main()