# -----------------------------------------
# python modules
# -----------------------------------------
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *
from lib.rpn_util import *
from lib.core import *

train_folder = '/home/garrick/Desktop/detective/data/kitti_split1/validation'
write_folder = '/home/garrick/Desktop/kitti_vis'

mkdir_if_missing(write_folder, delete_if_exist=True)

ann_folder = os.path.join(train_folder, 'label_2', '')
cal_folder = os.path.join(train_folder, 'calib', '')
im_folder = os.path.join(train_folder, 'image_2', '')

# get sorted filepaths
annlist = sorted(glob(ann_folder + '*.txt'))

imdb_start = time()

p2_last = None

for annind, annpath in enumerate(annlist):

    # get file parts
    base = os.path.basename(annpath)
    id, ext = os.path.splitext(base)

    calpath = os.path.join(cal_folder, id + '.txt')
    impath = os.path.join(im_folder, id + '.png')
    write_path = os.path.join(write_folder, id + '.jpg')

    im = cv2.imread(impath)

    # read gts
    p2 = read_kitti_cal(calpath)
    gts = read_kitti_label(annpath, p2)

    for gt in gts:

        if gt.cls not in ['Car']: continue

        display_str = '{:.8}'.format(gt.bbox_3d[10])

        #draw_text(im, display_str, gt.bbox_full)
        #draw_2d_box(im, gt.bbox_full, thickness=2)

        cx3d = gt.center_3d[0]
        cy3d = gt.center_3d[1]
        cz3d = gt.center_3d[2]

        cx2d = gt.bbox_3d[0]
        cy2d = gt.bbox_3d[1]
        cz2d = gt.bbox_3d[2]

        coord3d = np.linalg.inv(p2).dot(np.array([cx2d*cz2d, cy2d*cz2d, cz2d, 1]))

        cx3d = coord3d[0]
        cy3d = coord3d[1]
        cz3d = coord3d[2]

        w3d = gt.bbox_3d[3]
        h3d = gt.bbox_3d[4]
        l3d = gt.bbox_3d[5]
        rotY = gt.bbox_3d[-1]
        alpha = gt.bbox_3d[6]

        rotY = convertAlpha2Rot(alpha, cz3d, cx3d)

        verts3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY)
        draw_3d_box(im, verts3d, thickness=2)

    imwrite(im, write_path)

print('done')