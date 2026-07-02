import os
import sys
import torch
import numpy as np
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class PointEstimator():
    def __init__(self):
        self.model = importlib.import_module('pointnet2_reg_msg')
        self.classifier = self.model.get_model(out_channels=128)
        self.classifier.apply(inplace_relu)
        ckpt_path = os.path.join(BASE_DIR, 'log', 'regression_QM7', 'best_model.pth')
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.cuda().eval()

    def infer(self, pcl, centroid, m):
        with torch.no_grad():
            model_eval = self.classifier.eval()
            pcl = torch.tensor(pcl).cuda()
            pcl = pcl.unsqueeze(0).float()
            preds = model_eval(pcl)

        preds = np.array(preds.cpu())
        origin_dist = preds[0, 3] * m
        origin_center = -preds[0, :3] * m + centroid
        return origin_center, origin_dist
