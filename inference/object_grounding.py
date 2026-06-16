import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch

from .dep2pos import pixel_to_world_RGB
from sam2_simulator_predictor import SAM2_SIM_Predictor
from .dep2pcl import dep2pcl_func
from grounding_dino_predictor import grounding_dino
from Pointnet_Pointnet2_pytorch.infer import PointEstimator

from PIL import Image


class Ball():
    def __init__(self):
        self.x_ = None
        self.y_ = None
        self.z_ = None

        self.sam2_model = SAM2_SIM_Predictor()
        self.first_time = True
        self.pointnet = PointEstimator()
    
    def get_position(self,rgb_image, dep_image, text_prompt=None, bbox_prompt=None, point_prompt=None):
        # grounding dino
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = Image.fromarray(image_rgb)

        if self.first_time:
            self.first_time = False

            if text_prompt is not None:
                bbox = grounding_dino(rgb_image, text_prompt)
                mask = self.sam2_model.infer(image_rgb, bbox)
                
            else:
                if bbox_prompt is not None:
                    mask = self.sam2_model.infer(image_rgb, bbox_prompt)
                else:
                    mask = self.sam2_model.infer(image_rgb, point_prompt)

        elif text_prompt is not None:
            bbox = grounding_dino(rgb_image, text_prompt)
            self.sam2_model.if_init=False
            mask = self.sam2_model.infer(image_rgb, bbox)
        else:
            mask = self.sam2_model.infer(image_rgb)

        # mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
        
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)


        intrinsic = np.array([[604.716, 0, 322.682],[0, 603.582, 244.672],[0,  0,  1.0]])

        dep_image = mask*dep_image
        if dep_image.max()<=0:
            return None, None
        # print(mask.max())
        pcl, centroid, m = dep2pcl_func(dep_image, intrinsic)
        
        r=[
            [-0.4059,-0.65718,0.63509724],
            [ -0.90925742, 0.36045,  -0.208],
            [-0.09213272,  -0.66195527, -0.74385938],
        ]
        t = np.array([-0.900, 0.386, 0.466457])

        centroid_world = np.dot(r, centroid) + t
       

        #infer pos
        origin_center, origin_dist = self.pointnet.infer(pcl,centroid, m)

        world_center = np.dot(r, origin_center) + t
        print(world_center)
        return world_center*1000, origin_dist



        
        
        