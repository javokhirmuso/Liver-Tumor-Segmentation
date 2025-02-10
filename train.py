from roboflow import Roboflow
import torch
from ultralytics import YOLO

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="")
project = rf.workspace("studentdatasets").project("microscopy-cell-segmentation")
version = project.version(15)
dataset = version.download("yolov11")

# Load the YOLOv8 model
model = YOLO('yolo11n-seg.pt')  # Load the YOLO model

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Train the model
model.train(
    data='data path',  # Path to dataset config
    epochs=500,  # Number of epochs
    imgsz=640,  # Image size
    batch=8,  # Batch size
    workers=4,  # Number of data loading workers
    device=device,  # Specify device
    name='cell',  # Name of the training run
    optimizer='Adam',  # Optimizer type
    lr0=0.001,  # Initial learning rate
    weight_decay=5e-4,  # Weight decay
    patience=10  # Early stopping patience
)
