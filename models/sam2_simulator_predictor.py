import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

import sys
# sys.path.append('/home/nvidia/simulators/OmniGibson/omnigibson/segment_anything_2_real_time/demo')


class SAM2_SIM_Predictor():
    def __init__(self):

        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # from segment_anything_2_real_time import sam2
        # from segment_anything_2_real_time.sam2 import build_sam
        from sam2.build_sam import build_sam2_camera_predictor
        import time


        self.sam2_checkpoint = "/home/nvidia/simulators/OmniGibson/segment_anything_2_real_time/checkpoints/sam2_hiera_small.pt"
        # self.model_cfg = "/home/nvidia/simulators/OmniGibson/segment_anything_2_real_time/sam2_configs/sam2_hiera_s.yaml"
        self.model_cfg = "sam2_hiera_s.yaml"

        self.predictor = build_sam2_camera_predictor(self.model_cfg, self.sam2_checkpoint)
        self.if_init = False

        self.idx1 = 0

    def infer(self, frame, bbox=None):


        width, height = frame.shape[:2][::-1]
        if not self.if_init:

            self.predictor.load_first_frame(frame)
            self.if_init = True

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
           
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
            )

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                )

                all_mask = cv2.bitwise_or(all_mask, out_mask)

        else:
            out_obj_ids, out_mask_logits = self.predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                )

                all_mask = cv2.bitwise_or(all_mask, out_mask)

        return all_mask

