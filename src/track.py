import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def integrate_deep_sort(model, video_path, output_path, class_names, conf_threshold=0.5, nms_threshold=0.4):
    """
    Integrate Deep SORT for multi-object tracking.
    Args:
        model: Trained YOLOv8 model.
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        class_names (list): List of class names.
        conf_threshold (float): Confidence threshold for object detection.
        nms_threshold (float): Non-maximum suppression threshold.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
        bboxes = []
        scores = []
        for det in detections:
            class_id = int(det[5])
            if class_id in class_names:
                bbox = det[:4]
                score = det[4]
                bboxes.append(bbox)
                scores.append(score)

        tracks = deepsort.update_tracks(bboxes, scores, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = track.class_id
            label = f"{class_names[class_id]} {track_id}"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

def maintain_consistent_ids(video_path, output_path, class_names, conf_threshold=0.5, nms_threshold=0.4):
    """
    Maintain consistent IDs for multiple objects across video frames.
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        class_names (list): List of class names.
        conf_threshold (float): Confidence threshold for object detection.
        nms_threshold (float): Non-maximum suppression threshold.
    """
    model = YOLO('path/to/trained_model.pt')
    integrate_deep_sort(model, video_path, output_path, class_names, conf_threshold, nms_threshold)
