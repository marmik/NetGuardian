# NetGuardian

## SkySentinel: AI-powered UAV Vision System

SkySentinel is an AI-powered UAV vision system designed for real-time object detection and multi-object tracking using the VisDrone2019 dataset. The system leverages the YOLOv8 model for object detection and Deep SORT for tracking, providing accurate and efficient performance. The model is optimized for deployment on NVIDIA Jetson Nano, enabling real-time processing on edge devices.

### Features
- Real-time object detection using YOLOv8
- Multi-object tracking with Deep SORT
- Hyperparameter tuning with Optuna
- Model evaluation using mAP, precision, and recall
- Visualization of evaluation results
- Export to ONNX format and optimization with TensorRT
- Deployment on NVIDIA Jetson Nano
- Real-time video input/output with OpenCV
- Support for RTSP streams (optional)

## Setup

### Environment and Dependencies

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/marmik/NetGuardian.git
   cd NetGuardian
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training and Evaluation Scripts

To train the YOLOv8n model and evaluate its performance, follow these steps:

1. Load and preprocess the VisDrone2019 dataset:
   ```bash
   python src/data_loader.py
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```

3. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```

### Deploying the Model on NVIDIA Jetson Nano

To deploy the trained model on an NVIDIA Jetson Nano, follow these steps:

1. Export the model to ONNX format:
   ```bash
   python src/deploy.py --export
   ```

2. Optimize the model with TensorRT:
   ```bash
   python src/deploy.py --optimize
   ```

3. Deploy the model:
   ```bash
   python src/deploy.py --deploy
   ```

4. Run the real-time video input/output:
   ```bash
   python src/video_io.py
   ```
