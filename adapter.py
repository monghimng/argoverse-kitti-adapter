# Adapter 
"""The code to translate Argoverse dataset to KITTI dataset format"""

# Argoverse-to-KITTI Adapter

# Author: Yiyang Zhou 
# Email: yiyang.zhou@berkeley.edu
FILE_NAME_LEN = 9
DONT_CARE = 'DontCare'
print('\nLoading files...')

import json
import math
import os
from shutil import copyfile
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import pygame
import pyntcloud
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils import calibration

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

def construct_calib_str(calibration_data, pts_in_cam_coord=True):
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

def replace_nan_by_closest(Data):
    """
    From https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value.

    Replace any nan in the array by the first occurrence to the right, or to the left if none exists.
    Args:
        Data (): Array of interest. NOTE WE ONLY TESTED WITH 1D AND 2D ARRAY.

    Returns:
        The same array but with nan replaced.
    """
    nansIndx = np.where(np.isnan(Data))[0]
    isanIndx = np.where(~np.isnan(Data))[0]
    for nan in nansIndx:
        replacementCandidates = np.where(isanIndx>nan)[0]
        if replacementCandidates.size != 0:
            replacement = Data[isanIndx[replacementCandidates[0]]]
        else:
            replacement = Data[isanIndx[np.where(isanIndx<nan)[0][-1]]]
        Data[nan] = replacement
    return Data

def has_nan(arr):
    return np.isnan(arr).any()

from scipy.spatial import Delaunay

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

# testing coding for in_hull
# bbox = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 1],
#     [1, 0],
# ])
# pts = np.array([
#     [0.5, 0.5],
#     [2, 0.5]
# ])
# in_hull(pts, bbox)

def has_pts_in_bbox(pts, bbox, threshold=1):
    return in_hull(pts, bbox).sum() >= threshold


def get_bev(argoverse_data, argoverse_map, log_index, frame_index, bnds, meter_per_pixel, drivable_save_path,
            wanted_classes, wanted_save_paths, fov_save_path, visualize=True, filter_obj_if_no_pt_bbox=True,
            camera_calib=None):
    # output bev image setting
    bnd_width = bnds[1] - bnds[0]
    bnd_height = bnds[3] - bnds[2]
    bev_px_width = bnd_width / meter_per_pixel
    bev_px_height = bnd_height / meter_per_pixel
    if bev_px_width != 400 or bev_px_height != 400:
        print(bev_px_width, bev_px_height)

    # intput argoverse log and frame setting
    city_name = argoverse_data.city_name
    se3 = argoverse_data.get_pose(frame_index)
    x, y, _ = argoverse_data.get_pose(frame_index).translation
    bnds = bnds + np.array([x, x, y, y])
    polys = argoverse_map.find_local_driveable_areas(bnds, city_name)

    bev_drivable = pygame.Surface([bev_px_height, bev_px_width])
    bev_all = pygame.Surface([bev_px_height, bev_px_width])
    bev_fov = pygame.Surface([bev_px_height, bev_px_width])

    ############################## construct the fov mask (pixel=1 for bev pixels within camera fov)
    camera_img_width = 1920

    # pick 4 points in the image coordinate also with the depths to define the 4 corners of the fov
    # uv_depth, where u is the width of the image, v is height of image, depth is how far that obj is
    fov_corners_uv_depth = np.array([
        [0, 0, 0.01],  # upper left hand corner of the image, at depth 0.01 meter
        [0, 0, 100],  # upper left hand corner of the image, at depth 100 meter
        [camera_img_width - 1, 0, 100],  # upper right hand corner of the image, at depth 100 meter
        [camera_img_width - 1, 0, 0.01],  # upper right hand corner of the image, at depth 0.01 meter
        #         [960, 600, 100],
    ])
    calib = argoverse_data.get_calibration('ring_front_center')
    fov_corners_ego = calib.project_image_to_ego(fov_corners_uv_depth)

    # optionally transform to camera coords, this is required when we use
    # all 7 cameras. To make this code work with the rest of the code, we need to
    # manually change the axes because cam coord uses different rotation and axes orientation.
    # Refer to the Argoverse paper for the diagram on orientations of all coord system.
    if camera_calib:
        cam = camera_calib.project_ego_to_cam(fov_corners_ego)
        cam = cam[:, [2, 0, 1]]  # tranpose the xyz
        cam[:, [1, 2]] *= -1  # flip 2 of the axes
        fov_corners_ego = cam

    # project onto BEV by discarding the z dim (height)
    fov_corners_ego = fov_corners_ego[:,:2]

    # in the ego coord, +x is going forward and +y is to the left, but in BEV image
    # we expect +x (increment in rows) is going down and +y is to the right.
    # Therefore, we flip both coordinate
    fov_corners_ego *= -1

    # move origin to the center of the image
    fov_corners_ego += np.array(bnd_height // 2, bnd_width)

    # now in the coordinate sys of bev pixels
    fov_corners_bev = fov_corners_ego / meter_per_pixel

    # draw on the surface
    pygame.draw.polygon(bev_fov, [255, 0, 0], fov_corners_bev)

    ############################## drawing the drivable regions
    for poly_city in polys[1:]:

        # for some reason, the height dimension of the polygons sometimes might have NaN values
        # we simply replace them by the cloest values to the right
        if has_nan(poly_city):
            poly_city[:, 2] = replace_nan_by_closest(poly_city[:, 2])

        # transform the polygons from the city coordinate to the ego vehicle coordinate
        poly_ego = se3.inverse_transform_point_cloud(poly_city)

        # Refer to above for comments
        if camera_calib:
            poly_cam = camera_calib.project_ego_to_cam(poly_ego)
            poly_cam = poly_cam[:, [2, 0, 1]]  # tranpose the xyz
            poly_cam[:, [1, 2]] *= -1  # flip 2 of the axes
            poly_ego = poly_cam

        # project to 2d plane, now of shape [n, 2]
        poly_projected = np.array(poly_ego[:, :2])

        # in the ego coord, +x is going forward and +y is to the left, but in BEV image
        # we expect +x (increment in rows) is going down and +y is to the right.
        # Therefore, we flip both coordinate
        poly_projected *= -1

        # move origin to the center of the image
        poly_centered = poly_projected + np.array(bnd_height // 2, bnd_width)

        # map the polygon to pixel coordinate on the image so that we can draw it
        pixels = poly_centered / meter_per_pixel

        # note the color only matters for visualization
        pygame.draw.polygon(bev_drivable, [255, 0, 0], pixels)
        pygame.draw.polygon(bev_all, [255, 0, 0], pixels)

    ############################## drawing the objects of each classes

    # get the objects of a frame
    object_records_all = argoverse_data.get_label_object(frame_index)

    # filter out objects that no lidar points lie inside its bbox
    if filter_obj_if_no_pt_bbox:
        pts = argoverse_data.get_lidar(frame_index)
        object_records_all_filtered = [r for r in object_records_all if has_pts_in_bbox(pts, r.as_3d_bbox())]
        # print(len(object_records_all_filtered) / len(object_records_all))
        object_records_all = object_records_all_filtered

    bev_classes = []  # object class name mapped to the surface of that class
    for object_cls in wanted_classes:

        bev = pygame.Surface([bev_px_height, bev_px_width])

        # filter out other classes
        object_records = [r for r in object_records_all if r.label_class == object_cls]
        #         for r in object_records:
        #             if r.occlusion:
        #                 print(r.occlusion)

        # get 8 corners of each object
        # bboxes_corners is of shape [N, 8, 3] where N is the num of objects and 8 are the 8 corners
        bboxes_corners = np.array([r.as_3d_bbox() for r in object_records])

        # in case there were no object of that class, we still generate empty bev image
        if len(bboxes_corners) > 0:

            # determine the top 4 corners
            # from https://argoai.github.io/argoverse-api/argoverse.data_loading.html?highlight=corners#argoverse.data_loading.object_label_record.ObjectLabelRecord.as_3d_bbox,
            # we know the corners are ordered, and corner 0, 1, 5, 4 are the top 4 corners that would generate a polygon (IN THAT ORDER!)
            # we also remove the height dimension (which is the z dimension in xyz)
            corner_indices = 0, 1, 5, 4
            bboxes_4corners = bboxes_corners[:, corner_indices, :2]  # of size [N, 4, 2]

            # in the ego coord, +x is going up and +y is to the left, but in BEV image
            # we expect +x (increment in rows) is going down and +y is to the right.
            # Therefore, we flip both coordinate
            bboxes_4corners *= -1

            # move origin to the center of the image
            bboxes_4corners += np.array(bnd_height // 2, bnd_width)

            # map the polygon to pixel coordinate on the image so that we can draw it
            bboxes_4corners_pixels = bboxes_4corners / meter_per_pixel

            for bbox in bboxes_4corners_pixels:
                # note the color only matters for visualization
                pygame.draw.polygon(bev, [0, 255, 0], bbox)
                pygame.draw.polygon(bev_all, [0, 255, 0], bbox)

        bev_classes.append(bev)

    ############################## save to disk if paths are supplied

    # save drivable bev images to paths
    if drivable_save_path:
        # convert the pygame surface to binary array
        img_drivable = pygame.surfarray.array2d(bev_drivable)
        img_drivable[img_drivable != 0] = 1

        # flip 0 and 1 because the polygons
        # were actually given as NON-drivable region
        img_drivable = 1 - img_drivable

        cv2.imwrite(drivable_save_path, img_drivable)

    if fov_save_path:
        # convert the pygame surface to binary array
        img_fov = pygame.surfarray.array2d(bev_fov)
        img_fov[img_fov != 0] = 1
        cv2.imwrite(fov_save_path, img_fov)

    # save bev images to paths, for all wanted classes
    if wanted_save_paths:

        for bev_cls, cls_save_path in zip(bev_classes, wanted_save_paths):
            # convert the pygame surface to binary array
            img_cls = pygame.surfarray.array2d(bev_cls)
            img_cls[img_cls != 0] = 1

            cv2.imwrite(cls_save_path, img_cls)

    # visualize BEV outputs and camera images, useful when debugging in notebook
    if visualize:
        img = pygame.surfarray.array3d(bev_all)
        plt.figure()
        plt.imshow(img)

        img = pygame.surfarray.array3d(bev_fov)
        plt.figure()
        plt.imshow(img)

        import argoverse.visualization.visualization_utils as viz_util
        f, ax = viz_util.make_grid_ring_camera(argoverse_data, frame_index)
        plt.show()


def process_a_split(data_dir, target_data_dir, split_file_path, bev_bnds, bev_meter_per_pixel,
                    bev_wanted_classes, offset=0):
    """
    Args:
        data_dir: directory that contains data corresponding to A SPECIFIC
            SPLIT (train, validation, test)
        target_data_dir: the dir to write to. Contains data corresponding to
            A SPECIFIC SPLIT in KITTI (training, testing)
        split_file_path: location to write all the sample_idx of a split
        bev_bnds: [4] containing [x_min, x_max, y_min, y_max] in meter for the bev
        bev_meter_per_pixel: number of meters in a pixel in bev, most often fractional
        bev_wanted_classes: [VEHICLE, BICYCLIST, PEDESTRIAN, ...] the classes to be
            for bev
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

    ############################## for saving BEV segmentation masks paths
    target_bev_drivable_dir = os.path.join(target_data_dir, 'bev_DRIVABLE')
    os.makedirs(target_bev_drivable_dir, exist_ok=True)

    target_bev_fov_dir = os.path.join(target_data_dir, 'bev_FOV')
    os.makedirs(target_bev_fov_dir, exist_ok=True)

    target_bev_cls_dirs = []
    for wanted_cls in bev_wanted_classes:
        target_bev_cls_dir = os.path.join(target_data_dir, 'bev_{}'.format(wanted_cls))
        os.makedirs(target_bev_cls_dir, exist_ok=True)
        target_bev_cls_dirs.append(target_bev_cls_dir)
    ############################## end for saving BEV segmentation masks paths

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
    kitti_to_argo_mapping = {}
    for log_id in sorted(argoverse_loader.log_list):
        argoverse_data = argoverse_loader.get(log_id)
        from argoverse.map_representation.map_api import ArgoverseMap
        argoverse_map = ArgoverseMap()
        for cam in cams:

            ############################## Calibration for this whole log ##############################
            log_calib_path = os.path.join(data_dir, log_id, 'vehicle_calibration_info.json')
            calibration_data = calibration.load_calib(log_calib_path)[cam]
            calib_file_content = construct_calib_str(calibration_data)

            log_lidar_dir = os.path.join(data_dir, log_id, 'lidar')

            # Loop through the each lidar frame (10Hz) to rename, copy, and adapt
            # all images, lidars, calibration files, and label files.
            for frame_idx, timestamp in enumerate(sorted(argoverse_data.lidar_timestamp_list)):

                idx = str(i + offset).zfill(9)

                # recording the mapping from kitti to argo
                # (log index, the lidar frame index) uniquely identify a sample
                kitti_to_argo_mapping[idx] = (log_id, frame_idx)

                i += 1
                if i < total_number:
                    bar.update(i + 1)

                ############################## Lidar ##############################

                # Save lidar file into .bin format under the new directory
                lidar_file_path = os.path.join(log_lidar_dir, 'PC_{}.ply'.format(str(timestamp)))
                target_lidar_file_path = os.path.join(target_velodyne_dir, idx + '.bin')

                lidar_data = load_ply(lidar_file_path)
                lidar_data_augmented = np.concatenate((lidar_data, np.zeros([lidar_data.shape[0], 1])), axis=1)
                lidar_data_augmented = lidar_data_augmented.astype('float32')
                lidar_data_augmented.tofile(target_lidar_file_path)

                ############################## Image ##############################

                # Save the image file into .png format under the new directory
                cam_file_path = argoverse_data.image_list_sync[cam][frame_idx]
                target_cam_file_path = os.path.join(target_image_2_dir, idx + '.png')
                copyfile(cam_file_path, target_cam_file_path)

                target_calib_file_path = os.path.join(target_calib_dir, idx + '.txt')
                file = open(target_calib_file_path, 'w+')
                file.write(calib_file_content)
                file.close()

                ############################## BEV binary masks ##############################
                bev_drivable_save_path = os.path.join(target_bev_drivable_dir, idx + '.png')
                bev_wanted_save_paths = [os.path.join(cls_dir, idx + '.png') for cls_dir in target_bev_cls_dirs]
                bev_fov_save_path = os.path.join(target_bev_fov_dir, idx + '.png')

                get_bev(argoverse_data, argoverse_map, log_id, frame_idx, bev_bnds, bev_meter_per_pixel,
                        bev_drivable_save_path, bev_wanted_classes, bev_wanted_save_paths, bev_fov_save_path,
                        visualize=False, camera_calib=calibration_data)

                ############################## Labels ##############################

                if 'test' in split_file_path:
                    continue
                label_object_list = argoverse_data.get_label_object(frame_idx)
                target_label_2_file_path = os.path.join(target_label_2_dir, idx + '.txt')
                file = open(target_label_2_file_path, 'w+')

                # DontCare objects must appear at the end as per KITTI label files
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
                        # dz = p1[2] - p5[2]
                        # dx = p1[0] - p5[0]
                        # angle = math.atan2(dz, dx)

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

    # write kitti_to_argo_mapping to a file
    split_dir = os.path.dirname(split_file_path)
    split_name = os.path.basename(split_file_path)
    split_name = os.path.splitext(split_name)[0]
    prefix = 'kitti_to_argo_mapping'
    kitti_to_argo_mapping_path = os.path.join(split_dir, "{}_{}.json".format(prefix, split_name))
    file_handle = open(kitti_to_argo_mapping_path, 'w')
    json.dump(kitti_to_argo_mapping, file_handle)
    file_handle.close()

    bar.finish()
    print('Translation finished, processed {} files'.format(i))


if __name__ == '__main__':

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
    # the range of the BEV mask, in meters, in the coordinate sys of ego vehicle in argo
    # namely, x is forward facing wrt to the car, and y is sideway facing wrt to the car
    # we assume the difference max - min is even integer in meter in the following calculation
    bnd = 50.0
    bev_bnds = np.array([-bnd, +bnd, -bnd, +bnd], dtype=np.float32)
    bev_meter_per_pixel = 0.25 # resolution is 0.25 meter per pixel

    # the 7 object classes detailed in the cambridge paper, and the drivable region, we also do bicyclist here
    # because it is ood that cambridge paper only did bicycle
    bev_wanted_classes = [
        "BICYCLE",
        "BUS",
        "TRAILER",
        "MOTORCYCLIST",
        "LARGE_VEHICLE",
        "VEHICLE",
        "PEDESTRIAN",
        "BICYCLIST",
    ]

    root_dir = '/data/ck/data/argoverse/argoverse-tracking'
    target_dir = '/data/ck/data/argoverse/argoverse-tracking-kitti-format2'
    split_pairs = {
        'train': 'training',
        'val': 'training',  # in KITTI, validation data is also in training
        'test': 'testing'
    }
    image_set_dir = '/data/ck/data/argoverse/argoverse-tracking-kitti-format2/ImageSets'

    # for local testing
    root_dir = '/Users/ck/data_local/argo/argoverse-tracking'
    target_dir = '/Users/ck/data_local/argo/argoverse-tracking-kitti-format_cameracoord'
    split_pairs = {
        'sample': 'sample',
    }
    image_set_dir = os.path.join(target_dir, 'ImageSets')

    # # for local testing using the whole dataset
    # root_dir = '/Volumes/CK/data/argoverse/argoverse-tracking'
    # target_dir = '/Volumes/CK/data/argoverse/argoverse-tracking-kitti-format'
    # split_pairs = {
    #     'train': 'training',
    # }
    # image_set_dir = '/Volumes/CK/data/argoverse//argoverse-tracking-kitti-format/ImageSets'

    for k, (src_split, target_split) in enumerate(split_pairs.items()):
        data_dir = os.path.join(root_dir, src_split)
        target_data_dir = os.path.join(target_dir, target_split)
        split_file_path = os.path.join(image_set_dir, src_split + '.txt')
        offset = k * 100000000
        process_a_split(data_dir, target_data_dir, split_file_path, bev_bnds,
                        bev_meter_per_pixel, bev_wanted_classes, offset=offset)

"""
rm -r ~/data_local/argo/argoverse-tracking-kitti-format/; python ~/BEVSEG/argoverse-kitti-adapter/adapter.py
rm -r /data/ck/data/argoverse/argoverse-tracking-kitti-format/; python ~/BEVSEG/argoverse-kitti-adapter/adapter.py
python ~/BEVSEG/argoverse-kitti-adapter/adapter.py
"""
