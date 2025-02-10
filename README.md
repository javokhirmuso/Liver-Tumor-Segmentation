# Microscopy-cell-segmentation


## Dataset

- Source: Roboflow Universe
- url: https://universe.roboflow.com/studentdatasets/microscopy-cell-segmentation/dataset/15
- Format: YOLO format
- Classes: ['blood_vessel', 'glomerulus', 'unsure']

## Models Used

- YOLOv11n-seg.pt

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/javokhirmuso/Microscopy-cell-segmentation--.git
   cd Microscopy-cell-segmentation--
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using the Roboflow API:
   Modify the dataset path in the provided code as necessary.

## Usage

### Training

To train the model with your dataset:

```bash
python train.py 
```

### Inference

To run inference and detect diseases in images:

```bash
python detect.py --weights runs/segment/cell/best.pt --img 640 --conf 0.25 --source data/images/
```

## Metrics

- **defects label**
  <img src="runs\segment\cell\results.png" height="350px" width="100%"
        style="object-fit:contain"
    />

## Results

- **defects pred**
  <img src="runs\segment\cell\val_batch2_pred.jpg" height="500px" width="70%"
        style="object-fit:contain"
    />
- **defects label**
  <img src="runs\segment\cell\val_batch2_labels.jpg" height="500px" width="70%"
        style="object-fit:contain"
    />
