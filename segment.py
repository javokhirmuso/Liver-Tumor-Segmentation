from ultralytics import YOLO
import cv2
import numpy as np
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Object Detection and Segmentation')
    parser.add_argument('--source', type=str, default='0', help='source (video file or webcam number)')
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='model weights path')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--save', action='store_true', help='save results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load YOLO model
    model = YOLO(args.weights)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Video capture
    if args.source.isnumeric():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    # Video writer setup if saving is enabled
    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, 
                            (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference
        results = model(frame, conf=args.conf, device=device)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Display results
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # Save results if enabled
        if args.save:
            out.write(annotated_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()