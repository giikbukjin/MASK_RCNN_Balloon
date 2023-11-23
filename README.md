# Mask R-CNN for Object Detection and Segmentation
### 2023-2 OpenSourceSW Assignment
---

## 1. Loading the Dataset
### 1-1. MASK RCNN 공식 Github Clone
```
git clone https://github.com/matterport/Mask_RCNN.git
```

### 1-2. balloon_dataset 이미지 다운
[Download Link](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)

`Mask_RCNN/samples/balloon` 폴더에 저장

<br>

## 2. Data Labeling (with Labelme)
balloon.py가 있는 폴더로 이동
```
pip install labelme
labelme
```
labelme 이용해 balloon dataset labeling.

train 이미지를 라벨링한 json 파일은 train 폴더에, val 이미지를 라벨링한 json 파일은 val 폴더에 각각 저장

<br>

## 3. Training
### 3-1. balloon.py 구동
```
%cd /Users/.../Mask_RCNN/samples/balloon

python balloon.py train --dataset='/Users/.../Mask_RCNN/samples/balloon/balloon' --weights=coco
```

### 3-2. `inspect_balloon_model.ipynb` 수정
```
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("/Users/.../Mask_RCNN") # 경로 수정

...

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "/Users/.../Mask_RCNN/mask_rcnn_coco.h5"  # 경로 수정
config = balloon.BalloonConfig()

BALLOON_DIR = '/Users/.../Mask_RCNN/samples/balloon/balloon/' # 경로 수정
```

---
[Reference](https://hansonminlearning.tistory.com/16)
