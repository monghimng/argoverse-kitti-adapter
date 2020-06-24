# Adapter 
"""The code to translate Argoverse dataset to KITTI dataset format"""

# Argoverse-to-KITTI Adapter

# Author: Yiyang Zhou 
# Email: yiyang.zhou@berkeley.edu
FILE_NAME_LEN = 9
DONT_CARE = 'DontCare'
print('\nLoading files...')

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import os
from shutil import copyfile
from argoverse.utils import calibration
import json
import numpy as np
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
import math
import os
from typing import Union
import numpy as np
import pyntcloud
import progressbar
from time import sleep

"""
Your original file directory is:

argodataset
└── argoverse-tracking <----------------------------root_dir
    └── train <-------------------------------------data_dir
        └── 0ef28d5c-ae34-370b-99e7-6709e1c4b929
        └── 00c561b9-2057-358d-82c6-5b06d76cebcf
        └── ...
    └── validation
        └──5c251c22-11b2-3278-835c-0cf3cdee3f44
        └──...
    └── test
        └──8a15674a-ae5c-38e2-bc4b-f4156d384072
        └──...

Generate converted argo data in the exact KITTI directory structure:

argodataset
└── argoverse-tracking <----------------------------root_dir
    └──...
└── argoverse-tracking-kitti-format <----------------------------target_dir
    └── training
    └── testing
    └── ImageSets <----------------------------this is how most det model find out the splits
    
Unlike the 6 digits sample_idx of each frame in KITTI, the converted argo
dataset will have 9 digits. Like KITTI, where training and validation data
are both stored under training, with ImageSets split files differentiating the
two, we will put both training and validation data under training. A leading
0 in the sample_idx indicates it is training data, while a leading 1 indicates
it is in the validation set. These indices will also be recorded in ImageSets.
"""


_PathLike = Union[str, "os.PathLike[str]"]
def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: Array of shape (N, 3). Note that Argoverse lidar data doesn't have reflectance. :(
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)

def write_split_file(split_file_path, offset, num_sample):
    """
    Generate indices in the range of [offset, offset + num_sample] and write them
    to a file.
    Args:
        split_file_path:
        offset:
        num_sample:

    Returns:

    """
    # first make sure its parent directory exist
    dir_name = os.path.dirname(split_file_path)
    os.makedirs(dir_name, exist_ok=True)

    # generate the indices
    indices = list(range(offset, offset + num_sample))
    indices = [str(i).zfill(FILE_NAME_LEN) for i in indices]
    s = '\n'.join(indices)

    # write to file
    with open(split_file_path, 'w')as f:
        f.write(s)

def construct_calib_str(calibration_data):
    """
    convert the argo calib to KITTI format, and return the kitti calib as a string.
    Args:
        calibration_data: an object of class argoverse.utils.calibration.Calibration
    """
    L3 = 'P2: '
    for j in calibration_data.K.reshape(1, 12)[0]:
        L3 = L3 + str(j) + ' '
    L3 = L3[:-1]

    L6 = 'Tr_velo_to_cam: '
    for k in calibration_data.extrinsic.reshape(1, 16)[0][0:12]:
        L6 = L6 + str(k) + ' '
    L6 = L6[:-1]

    L1 = 'P0: 0 0 0 0 0 0 0 0 0 0 0 0'
    L2 = 'P1: 0 0 0 0 0 0 0 0 0 0 0 0'
    L4 = 'P3: 0 0 0 0 0 0 0 0 0 0 0 0'
    L5 = 'R0_rect: 1 0 0 0 1 0 0 0 1'
    L7 = 'Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0'

    file_content = "\n".join([L1, L2, L3, L4, L5, L6, L7])
    return file_content

def process_a_split(data_dir, target_data_dir, split_file_path, offset=0):
    """
    Args:
        data_dir: directory that contains data corresponding to A SPECIFIC
            SPLIT (train, validation, test)
        target_data_dir: the dir to write to. Contains data corresponding to
            A SPECIFIC SPLIT in KITTI (training, testing)
        split_file_path: location to write all the sample_idx of a split
        offset: first sample_idx to start counting from, needed to differentiate
            training and validation sample_idx because in KITTI they are under
            the same dir
    """
    print("Processing", data_dir)

    target_velodyne_dir = os.path.join(target_data_dir, 'velodyne')
    target_velodyne_reduced_dir = os.path.join(target_data_dir, 'velodyne_reduced')
    target_calib_dir = os.path.join(target_data_dir, 'calib')
    target_image_2_dir = os.path.join(target_data_dir, 'image_2')
    if 'test' not in split_file_path:
        target_label_2_dir = os.path.join(target_data_dir, 'label_2')

    os.makedirs(target_velodyne_dir, exist_ok=True)
    os.makedirs(target_velodyne_reduced_dir, exist_ok=True)
    os.makedirs(target_calib_dir, exist_ok=True)
    os.makedirs(target_image_2_dir, exist_ok=True)
    if 'test' not in split_file_path:
        os.makedirs(target_label_2_dir, exist_ok=True)

    # Check the number of logs, defined as one continuous trajectory
    argoverse_loader = ArgoverseTrackingLoader(data_dir)
    print('Total number of logs:', len(argoverse_loader))
    argoverse_loader.print_all()
    print('\n')

    cams = [
        'ring_front_center',
        # 'ring_front_left',
        # 'ring_front_right',
        # 'ring_rear_left',
        # 'ring_rear_right',
        # 'ring_side_left',
        # 'ring_side_right'
    ]

    # count total number of files
    total_number = 0
    for q in argoverse_loader.log_list:
        log_lidar_path = os.path.join(data_dir, q, 'lidar')
        path, dirs, files = next(os.walk(log_lidar_path))
        total_number = total_number + len(files)

    total_number = total_number * len(cams)

    print('Now write sample indices to split file.')
    write_split_file(split_file_path, offset, total_number)

    bar = progressbar.ProgressBar(maxval=total_number, \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    print('Total number of files: {}. Translation starts...'.format(total_number))
    print('Progress:')
    bar.start()

    i = 0
    for log_id in sorted(argoverse_loader.log_list):
        argoverse_data = argoverse_loader.get(log_id)
        for cam in cams:
            # Recreate the calibration file content
            log_calib_path = os.path.join(data_dir, log_id, 'vehicle_calibration_info.json')
            calibration_data = calibration.load_calib(log_calib_path)[cam]
            calib_file_content = construct_calib_str(calibration_data)

            l = 0

            log_lidar_dir = os.path.join(data_dir, log_id, 'lidar')

            # Loop through the each lidar frame (10Hz) to copy and reconfigure all images, lidars, calibration files, and label files.
            for timestamp in argoverse_data.lidar_timestamp_list:

                # import pdb;pdb.set_trace()
                idx = str(i + offset).zfill(9)
                i += 1
                if i < total_number:
                    bar.update(i + 1)

                # Save lidar file into .bin format under the new directory
                lidar_file_path = os.path.join(log_lidar_dir, 'PC_{}.ply'.format(str(timestamp)))
                target_lidar_file_path = os.path.join(target_velodyne_dir, idx + '.bin')

                lidar_data = load_ply(lidar_file_path)
                lidar_data_augmented = np.concatenate((lidar_data, np.zeros([lidar_data.shape[0], 1])), axis=1)
                lidar_data_augmented = lidar_data_augmented.astype('float32')
                lidar_data_augmented.tofile(target_lidar_file_path)

                # Save the image file into .png format under the new directory
                cam_file_path = argoverse_data.image_list_sync[cam][l]
                target_cam_file_path = os.path.join(target_image_2_dir, idx + '.png')
                copyfile(cam_file_path, target_cam_file_path)

                target_calib_file_path = os.path.join(target_calib_dir, idx + '.txt')
                file = open(target_calib_file_path, 'w+')
                file.write(calib_file_content)
                file.close()

                if 'test' in split_file_path:
                    continue
                label_object_list = argoverse_data.get_label_object(l)
                target_label_2_file_path = os.path.join(target_label_2_dir, idx + '.txt')
                file = open(target_label_2_file_path, 'w+')
                l += 1

                # DontCare objects must appear at the end in KITTI
                object_lines = []
                dontcare_lines = []

                for detected_object in label_object_list:
                    classes = detected_object.label_class
                    classes = OBJ_CLASS_MAPPING_DICT[classes] # map class type from artoverse to KITTI
                    occulusion = round(detected_object.occlusion / 25)
                    height = detected_object.height
                    length = detected_object.length
                    width = detected_object.width
                    truncated = 0

                    center = detected_object.translation  # in ego frame

                    corners_ego_frame = detected_object.as_3d_bbox()  # all eight points in ego frame
                    corners_cam_frame = calibration_data.project_ego_to_cam(
                        corners_ego_frame)  # all eight points in the camera frame
                    image_corners = calibration_data.project_ego_to_image(corners_ego_frame)
                    image_bbox = [min(image_corners[:, 0]), min(image_corners[:, 1]), max(image_corners[:, 0]),
                                  max(image_corners[:, 1])]
                    # the four coordinates we need for KITTI
                    image_bbox = [round(x) for x in image_bbox]

                    center_cam_frame = calibration_data.project_ego_to_cam(np.array([center]))

                    # if 0 < center_cam_frame[0][2] < max_d and 0 < image_bbox[0] < 1920 and 0 < image_bbox[1] < 1200 and 0 < \
                    #     image_bbox[2] < 1920 and 0 < image_bbox[3] < 1200:
                    if True:
                        # the center coordinates in cam frame we need for KITTI

                        # for the orientation, we choose point 1 and point 5 for application
                        p1 = corners_cam_frame[1]
                        p5 = corners_cam_frame[5]
                        dz = p1[2] - p5[2]
                        dx = p1[0] - p5[0]
                        angle = math.atan2(dz, dx)

                        # todo
                        angle_vec = p1 - p5
                        # norm vec along the x axis, points to the right in KITTI cam rect coordinate
                        origin_vec = np.array([1, 0, 0])
                        import vg
                        angle = vg.signed_angle(origin_vec, angle_vec, look=vg.basis.y, units='rad')

                        # the orientation angle of the car
                        beta = math.atan2(center_cam_frame[0][2], center_cam_frame[0][0])
                        alpha = angle + beta - math.pi / 2
                        line = classes + ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(round(truncated, 2),
                                                                                                occulusion, round(alpha, 2),
                                                                                                round(image_bbox[0], 2),
                                                                                                round(image_bbox[1], 2),
                                                                                                round(image_bbox[2], 2),
                                                                                                round(image_bbox[3], 2),
                                                                                                round(height, 2),
                                                                                                round(width, 2),
                                                                                                round(length, 2), round(
                                center_cam_frame[0][0], 2), round(center_cam_frame[0][1], 2) + height / 2, round(center_cam_frame[0][2],
                                                                                                    2), round(angle, 2))


                        # separate the object lines so we can move all the dontcare lines at the end
                        if classes == DONT_CARE:
                            dontcare_lines.append(line)
                        else:
                            object_lines.append(line)

                for line in object_lines:
                    file.write(line)
                for line in dontcare_lines:
                    file.write(line)
                file.close()

    bar.finish()
    print('Translation finished, processed {} files'.format(i))


####CONFIGURATION#################################################
# Root directory
root_dir = '/data/ck/data/argoverse/argoverse-tracking'
# Maximum thresholding distance for labelled objects
# (Object beyond this distance will not be labelled)
max_d = 50
split = "train"  # one of train, val, and test

# argoverse object class to KITTI
car = 'Car'
ped = 'Pedestrian'
cyc = 'Cyclist'

# essentially only keep standard sized car, pedestrians, and cyclist.
OBJ_CLASS_MAPPING_DICT = {
    "VEHICLE": car,
    "PEDESTRIAN": ped,
    "BICYCLIST": cyc,
    "EMERGENCY_VEHICLE": DONT_CARE,
    "BUS": DONT_CARE,
    "LARGE_VEHICLE": DONT_CARE,
    "ON_ROAD_OBSTACLE": DONT_CARE,
    "BICYCLE": DONT_CARE,
    "OTHER_MOVER": DONT_CARE,
    "TRAILER": DONT_CARE,
    "MOTORCYCLIST": DONT_CARE,
    "MOPED": DONT_CARE,
    "MOTORCYCLE": DONT_CARE,
    "STROLLER": DONT_CARE,
    "ANIMAL": DONT_CARE,
    "WHEELCHAIR": DONT_CARE,
    "SCHOOL_BUS": DONT_CARE,
}
####################################################################


root_dir = '/data/ck/data/argoverse/argoverse-tracking'
target_dir = '/data/ck/data/argoverse/argoverse-tracking-kitti-format'
split_pairs = {
    'train': 'training',
    'val': 'training',  # in KITTI, validation data is also in training
    'test': 'testing'
}
image_set_dir = '/data/ck/data/argoverse/argoverse-tracking-kitti-format/ImageSets'

# # for local testing
# root_dir = '/Users/ck/data_local/argo/argoverse-tracking'
# target_dir = '/Users/ck/data_local/argo/argoverse-tracking-kitti-format'
# split_pairs = {
#     'sample': 'sample',
# }
# image_set_dir = '/Users/ck/data_local/argo/argoverse-tracking-kitti-format/ImageSets'

for k, (src_split, target_split) in enumerate(split_pairs.items()):
    data_dir = os.path.join(root_dir, src_split)
    target_data_dir = os.path.join(target_dir, target_split)
    split_file_path = os.path.join(image_set_dir, src_split + '.txt')
    offset = k * 100000000
    process_a_split(data_dir, target_data_dir, split_file_path, offset)

"""
rm -r ~/data_local/argo/argoverse-tracking-kitti-format/; python ~/BEVSEG/argoverse-kitti-adapter/adapter.py
rm -r /data/ck/data/argoverse/argoverse-tracking-kitti-format/; python ~/BEVSEG/argoverse-kitti-adapter/adapter.py
"""
