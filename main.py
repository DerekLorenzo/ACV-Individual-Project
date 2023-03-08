# Author: Derek Lorenzo
# Class: COMS 4995 Sec 6
# Project: Individual Project - Object Detection Using Faster-RCNN
# templates and helper files provided by https://pytorch.org/tutorials/

import os
import sys
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
from engine import train_one_epoch, evaluate
import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('Qt5Agg')


# Generates the dataset that will be used to train and test the model.
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = list(sorted(os.listdir("./images")))
        self.annotations = list(sorted(os.listdir("./annotations")))

    def __getitem__(self, idx):
        image_path = os.path.join("images", self.images[idx])
        image = Image.open(image_path).convert("RGB")
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        label_path = os.path.join("annotations", self.annotations[idx])

        boxes = []
        labels = []
        area = []

        tree = ET.parse(label_path)
        root = tree.getroot()
        for box in root.iter('object'):
            xmin = int(box.find("bndbox/xmin").text)
            xmax = int(box.find("bndbox/xmax").text)
            ymin = int(box.find("bndbox/ymin").text)
            ymax = int(box.find("bndbox/ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((xmax - xmin) * (ymax - ymin))

            label = box.find("name").text
            if label == "with_mask":
                labels.append(1)
            elif label == "mask_weared_incorrect":
                labels.append(2)
            elif label == "without_mask":
                labels.append(3)
            else:
                labels.append(0)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target

    def __len__(self):
        return len(self.images)


# Returns a pre-trained Faster CNN model with default weights
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Displays image with bounding boxes labelled by color for whether the model predicted
# they were wearing a mask, incorrectly wearing a mask, or not wearing a mask
def visualize_predictions(images, predictions):
    for index in range(len(images)):
        image = images[index].to('cpu').data

        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))

        for box, label in zip(predictions[index]["boxes"], predictions[index]["labels"]):
            xmin, ymin, xmax, ymax = box.to('cpu')

            if label == 1:
                edgecolor = 'r'
            elif label == 2:
                edgecolor = 'g'
            elif label == 3:
                edgecolor = 'b'
            else:
                edgecolor = 'y'

            ax.add_patch(Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                   linewidth=1, edgecolor=edgecolor, facecolor='none'))

        plt.show()


# Generates predictions from a previously saved model. Useful if model has been previously
# fine-tuned and now looking to predict on a different dataset.
def predict():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model()
    model.load_state_dict(torch.load('model.pt'))
    model.to(device)
    dataset_test = MaskDataset()
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    coco_evaluator, images, predictions = evaluate(model, data_loader_test, device=device)
    visualize_predictions(images[:10], predictions[:10])


# Generates predictions by first organizing train and test datasets, then training
# the model, and finally visualizing the predictions. Model is saved to potentially
# be loaded at a later time.
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = MaskDataset()
    dataset_test = MaskDataset()
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 11

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), 'model.pt')

    coco_evaluator, images, predictions = evaluate(model, data_loader_test, device=device)
    visualize_predictions(images[:15], predictions[:15])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        predict()
    else:
        main()
