Author: Derek Lorenzo
Class: COMS 4995 Sec 6
Project: Individual Project - Object Detection Using Faster-RCNN

The purpose of this project is to fine-tune a Faster RCNN model
to detect whether individuals in photos are wearing a mask correctly,
wearing a mask incorrectly, or not wearing a mask. Such a model
may be useful in certain environments, such as hospitals, where
the utilization of masks is extraordinary importance.

This project utilizes libraries, helper files, and pre-trained models
from PyTorch and a dataset of 853 images from Kaggle.

Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
PyTorch: https://pytorch.org/

Training and testing the model is completed by running main.py. templates
and helper files utilized or called by main.py from PyTorch tutorials are
employed with modifications made where necessary. Helper files include
coco_eval.py, coco_utils.py, engine.py, transforms.py, utils.py.

PyTorch Tutorials: https://pytorch.org/tutorials/