from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class DroneDataset(MonoDataset):

    def __init__(self, *args, **kwargs):
        super(DroneDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.576, 0, 0.5, 0],
                           [0, 0.768, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1024, 768)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class Drone1Dataset(DroneDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(Drone1Dataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join( self.data_path, "uav")
        image_path = os.path.join(
            image_path,
            "{:02d}".format(int(folder)),
            f_str)
        return image_path

class Drone2Dataset(DroneDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(Drone2Dataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = str(frame_index)+self.img_ext
        image_path = os.path.join( self.data_path, "picture")
        image_path = os.path.join(
            image_path,
            str(folder),
            f_str)
        return image_path
