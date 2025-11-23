# Artifact-Resilient Real-Time Holography
### [Project Page](https://light.princeton.edu/publication/real-time-arh/) | [Paper](https://light.princeton.edu/wp-content/uploads/2025/10/ARH_main.pdf)


[Victor Chu](https://bkv2chu.github.io), [Oscar Pueyo-Ciutad](https://opueyociutad.github.io/),  [Ethan Tseng](https://ethan-tseng.github.io), [Florian Schiffers](https://florianschiffers.de/), [Grace Kuo](https://grace-kuo.com/), [Nathan Matsuda](https://www.nathanmatsuda.com/), [Albert Redo-Sanchez](https://redo-sanchez.github.io/), [Douglas Lanman](https://www.linkedin.com/in/dlanman/), [Oliver Cossairt](https://compphotolab.northwestern.edu/people/oliver-ollie-cossairt/), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

We introduces a novel method for generating artifact-resilient phase holograms in real-time. This code implements the following:
* A model for simulating pre and post-pupil obstructions (a common source of holographic artifacts).
* A differentiable metric to quantify holographic artifact resilience (see `rayleigh_distance_loss` in `holo_utils.py`).
* Training and inference of a real-time neural network to create pseudo-random phase holograms that are inherently artifact-resilient.

This code builds on [Neural-Holography](https://github.com/computational-imaging/neural-holography) and [Pado](https://github.com/shwbaek/pado) repositories.

## Installation

### 1. Create and Activate a Conda Environment

```bash
conda create -n arh python=3.10 -y
conda activate arh
```

### 2. Install Required Packages With pip


```bash
pip install uv

uv pip install -r requirements.txt
```


## Training RealTime ARH

Place your training data in the `data` directory. Then run the following commands: 

```bash
python train_ARH.py --channel=0 --run_id=experiment_red
python train_ARH.py --channel=1 --run_id=experiment_green
python train_ARH.py --channel=2 --run_id=experiment_blue
```

## Evaluating Pretrained Models

For evaluation, we provide four pretrained models for each color channel (R, G, and B). You can run inference using the provided scripts:

```bash
./ensemble_inference_2d.sh
./ensemble_inference_3d.sh
```

Our RGBD evaluation data comes from [Split Lohmann Multifocal Displays](https://github.com/Image-Science-Lab-cmu/SplitLohmann/tree/main/scenes).



## Citation
```
@article{Chu2025RealTime,
author = {Chu, Victor and Pueyo-Ciutad, Oscar and Tseng, Ethan and Schiffers, Florian and Kuo, Grace and Matsuda, Nathan and Redo-Sanchez, Albert and Lanman Douglas and Cossairt, Oliver and Heide, Felix},
title = {Artifact-Resilient Real-Time Holography},
year = {2025},
issue_date = {December 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3763361},
doi = {10.1145/3763361},
journal = {ACM Trans. Graph.},
month = dec,
articleno = {219},
numpages = {13}
}
```

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 
