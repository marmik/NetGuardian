import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml

def load_and_preprocess_visdrone(data_dir, output_dir):
    """
    Load and preprocess the VisDrone2019 dataset.
    Args:
        data_dir (str): Path to the VisDrone2019 dataset.
        output_dir (str): Path to save the preprocessed dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(data_dir, 'images')
    annot_dir = os.path.join(data_dir, 'annotations')

    for image_file in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        annot_path = os.path.join(annot_dir, image_file.replace('.jpg', '.xml'))

        if not os.path.exists(annot_path):
            continue

        tree = ET.parse(annot_path)
        root = tree.getroot()

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objects.append((name, xmin, ymin, xmax, ymax))

        output_image_path = os.path.join(output_dir, 'images', image_file)
        output_annot_path = os.path.join(output_dir, 'labels', image_file.replace('.jpg', '.txt'))

        if not os.path.exists(os.path.dirname(output_image_path)):
            os.makedirs(os.path.dirname(output_image_path))
        if not os.path.exists(os.path.dirname(output_annot_path)):
            os.makedirs(os.path.dirname(output_annot_path))

        shutil.copy(image_path, output_image_path)

        with open(output_annot_path, 'w') as f:
            for obj in objects:
                name, xmin, ymin, xmax, ymax = obj
                f.write(f"{name} {xmin} {ymin} {xmax} {ymax}\n")

def convert_to_yolov8_format(data_dir, output_dir, class_names):
    """
    Convert the dataset into YOLOv8-compatible format.
    Args:
        data_dir (str): Path to the preprocessed dataset.
        output_dir (str): Path to save the YOLOv8-compatible dataset.
        class_names (list): List of class names.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    for image_file in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            continue

        output_image_path = os.path.join(output_dir, 'images', image_file)
        output_label_path = os.path.join(output_dir, 'labels', image_file.replace('.jpg', '.txt'))

        if not os.path.exists(os.path.dirname(output_image_path)):
            os.makedirs(os.path.dirname(output_image_path))
        if not os.path.exists(os.path.dirname(output_label_path)):
            os.makedirs(os.path.dirname(output_label_path))

        shutil.copy(image_path, output_image_path)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        with open(output_label_path, 'w') as f:
            for line in lines:
                name, xmin, ymin, xmax, ymax = line.strip().split()
                class_id = class_names.index(name)
                x_center = (int(xmin) + int(xmax)) / 2.0
                y_center = (int(ymin) + int(ymax)) / 2.0
                width = int(xmax) - int(xmin)
                height = int(ymax) - int(ymin)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump({'train': os.path.join(output_dir, 'images'), 'val': os.path.join(output_dir, 'images'), 'nc': len(class_names), 'names': class_names}, f)
