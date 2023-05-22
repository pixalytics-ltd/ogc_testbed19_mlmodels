# OGC Testbed-19 Machine Learning models contribution

This repository contains content related to the Testbed 19 Machine Learning models task

The tests being performed by Pixalytics Ltd included:
* Installing and testing Meta SAM model with RGB and hyperspectral data (this repo)
* Testing plastics model for transfer learning (seperate repo)

# Installing environment and setting up SAM model

* Create basic environment: `conda env create`

* Install SAM model: `pip install git+https://github.com/facebookresearch/segment-anything.git`

* Download SAM weights: `wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

