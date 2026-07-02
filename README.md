# SCOOP'D Codebase

Codebase for **SCOOP'D: Learning Mixed-Liquid-Solid Scooping via Sim2Real Generative Policy**.

[[Project Page](https://scoopdiff.github.io/)] [[arXiv](https://arxiv.org/abs/2510.11566)]

SCOOP'D studies robotic scooping in mixed liquid-solid environments. The system collects scooping demonstrations in simulation, learns generative policies with Diffusion Policy, and deploys the learned policy to real-world robot scooping tasks.

> This repository is under active cleanup. Full installation instructions, checkpoints, datasets, and training scripts will be released later.

## Structure

```text
SCOOPD_Codebase/
├── simulator/      # OmniGibson simulation and data collection
├── models/         # Policy and perception models
├── inference/      # Real-world inference and robot execution
└── README.md
```

## Main Components

* `simulator/`: simulation environment and heuristic data collection.
* `models/diffusion_policy/`: Diffusion Policy dependency.
* `models/GroudingDINO/`: GroundingDINO dependency.
* `models/sam2/`: SAM2 dependency.
* `models/pointnet/`: PointNet++ object state regression.
* `inference/`: RGB-D perception, target localization, policy inference, and robot execution.

## Usage

Simulation data collection:

```bash
python simulator/omnigibson/examples/learning/collect_data_row.py
```

Real-world inference:

```bash
python inference/main_policy_axis_record.py
```

Before running, please configure local paths, checkpoints, camera topics, robot SDK, robot IP, and calibration files.

## Citation

```bibtex
@article{wang2025scoop,
  title={SCOOP'D: Learning Mixed-Liquid-Solid Scooping via Sim2Real Generative Policy},
  author={Wang, Kuanning and Gu, Yongchong and Fu, Yuqian and Shangguan, Zeyu and He, Sicheng and Xue, Xiangyang and Fu, Yanwei and Seita, Daniel},
  journal={arXiv preprint arXiv:2510.11566},
  year={2025}
}
```
