## Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
import shutil
import random
import subprocess
import sys
import yaml
from pathlib import Path
from PIL import Image
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve, mean_absolute_error

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

from pycocotools.coco import COCO



df = pd.read_csv('Malaria.csv')
df

# Image folder and annotations folder
#images_folder = 'images'
#annotations_folder = 'annotations'

## Mask RCNN with Det

# Paths to the dataset
train_images = "train/images"
train_annotations = "train/train_coco_annotations_fixed.json"
val_images = "val/images"
val_annotations = "val/val_coco_annotations_fixed.json"
test_images = "test/images"
test_annotations = "test/test_coco_annotations_fixed.json"

# Categories (optional metadata)
categories = [
    {"id": 1, "name": "RBC"},
    {"id": 2, "name": "LKC"},
    {"id": 3, "name": "TRP"},
    {"id": 4, "name": "SCH"},
    {"id": 5, "name": "RNG"},
    {"id": 6, "name": "GMC"},
]

# Register the datasets
register_coco_instances("train_set", {}, train_annotations, train_images)
register_coco_instances("val_set", {}, val_annotations, val_images)
register_coco_instances("test_set", {}, test_annotations, test_images)

# Set metadata
MetadataCatalog.get("train_set").set(thing_classes=[cat["name"] for cat in categories])
MetadataCatalog.get("val_set").set(thing_classes=[cat["name"] for cat in categories])
MetadataCatalog.get("test_set").set(thing_classes=[cat["name"] for cat in categories])

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set",)
cfg.DATASETS.TEST = ("val_set",)  # Use validation for evaluation during training
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)  # Number of categories
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on GPU memory
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 1000  # Adjust based on dataset size
cfg.SOLVER.STEPS = []  # Disable learning rate decay
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Number of RoIs per image
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.DEVICE = "cpu"

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Load the trained model for inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Threshold for predictions
predictor = DefaultPredictor(cfg)

# Evaluate on test set
test_metadata = MetadataCatalog.get("test_set")
test_dataset = DatasetCatalog.get("test_set")

y_true, y_pred, iou_scores = [], [], []
ground_truth_boxes, predicted_boxes = [], []
precisions, recalls = [], []
map_metric = MeanAveragePrecision()

for data in test_dataset:
    image = cv2.imread(data["file_name"])
    outputs = predictor(image)

    # Visualize predictions
    v = Visualizer(image[:, :, ::-1], metadata=test_metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
    cv2.waitKey(1)

    # Metrics calculation
    gt_boxes = np.array([ann["bbox"] for ann in data["annotations"]])
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    gt_labels = np.array([ann["category_id"] for ann in data["annotations"]])
    pred_labels = outputs["instances"].pred_classes.cpu().numpy()

    y_true.extend(gt_labels)
    y_pred.extend(pred_labels)
    ground_truth_boxes.extend(gt_boxes)
    predicted_boxes.extend(pred_boxes)

    # Compute IoU for each pair of GT and predicted boxes
    iou_scores.extend([
        np.max([box_iou(gt_box, pred_box) for pred_box in pred_boxes])
        for gt_box in gt_boxes
    ])

    # Compute mAP
    map_metric.update({
        "pred_boxes": [torch.tensor(pred_boxes)],
        "pred_labels": [torch.tensor(pred_labels)],
        "pred_scores": [outputs["instances"].scores.cpu()],
        "gt_boxes": [torch.tensor(gt_boxes)],
        "gt_labels": [torch.tensor(gt_labels)],
    })

# Calculate final metrics
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
mae = mean_absolute_error(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred, target_names=[cat["name"] for cat in categories])
mean_iou = np.mean(iou_scores)
map_result = map_metric.compute()

# Print metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"IoU: {mean_iou:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"mAP: {map_result['map']:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
plt.plot(recall_vals, precision_vals, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Save Training and Validation Loss Plot
def save_loss_plot(trainer, output_dir="./output"):
    training_loss = trainer.storage.history("total_loss")
    validation_loss = trainer.storage.history("validation_loss")
    epochs = range(len(training_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    
# Save Metrics to CSV
def save_metrics_to_csv(metrics, output_csv="./output/test_metrics.csv"):
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for metric, value in metrics.items():
            writer.writerow([metric, value])
            

# Save Precision-Recall Curve
def save_precision_recall_curve(y_true, y_pred, output_path="./output/pr_curve.png"):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    
# After Training
save_loss_plot(trainer)

# Evaluate on Test Set
metrics = {}
metrics["Precision"] = precision
metrics["Recall"] = recall
metrics["F1-Score"] = f1
metrics["IoU"] = mean_iou
metrics["Mean Absolute Error"] = mae
metrics["mAP"] = map_result["map"]

save_metrics_to_csv(metrics)

save_precision_recall_curve(y_true, y_pred)