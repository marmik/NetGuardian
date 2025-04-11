import os
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def evaluate_model(model_path, data_yaml, iou_threshold=0.5):
    """
    Evaluate the model using mAP, precision, and recall.
    Args:
        model_path (str): Path to the trained model.
        data_yaml (str): Path to the data.yaml file.
        iou_threshold (float): IoU threshold for evaluation.
    Returns:
        dict: Dictionary containing mAP, precision, and recall.
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, iou_thres=iou_threshold)
    
    metrics = {
        'mAP': results['metrics/mAP_0.5'],
        'precision': results['metrics/precision'],
        'recall': results['metrics/recall']
    }
    
    return metrics

def visualize_evaluation_results(metrics, output_dir):
    """
    Visualize the evaluation results.
    Args:
        metrics (dict): Dictionary containing mAP, precision, and recall.
        output_dir (str): Path to save the visualization results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(metrics['true_labels'], metrics['pred_scores'])
    average_precision = average_precision_score(metrics['true_labels'], metrics['pred_scores'])
    
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # Plot mAP, precision, and recall
    plt.figure()
    plt.bar(['mAP', 'Precision', 'Recall'], [metrics['mAP'], metrics['precision'], metrics['recall']])
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))

if __name__ == '__main__':
    model_path = 'path/to/trained_model.pt'
    data_yaml = 'path/to/data.yaml'
    output_dir = 'path/to/output_dir'
    
    metrics = evaluate_model(model_path, data_yaml)
    visualize_evaluation_results(metrics, output_dir)
