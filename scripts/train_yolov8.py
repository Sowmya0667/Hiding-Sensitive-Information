from ultralytics import YOLO
import argparse

def train_yolov8(data_path, model_name="yolov8n.pt", epochs=100, batch=4, workers=2, imgsz=416):
    """
    Train a YOLOv8 model for ID card detection.
    
    Args:
        data_path (str): Path to the data.yaml configuration file.
        model_name (str): Pre-trained YOLOv8 model to use (default: yolov8n.pt).
        epochs (int): Number of training epochs.
        batch (int): Batch size for training.
        workers (int): Number of data loader workers.
        imgsz (int): Image size for training.
    """
    # Initialize YOLO model
    model = YOLO(model_name)

    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        fliplr=0.0,        
        mosaic=0.0,        
        hsv_h=0.0,       
        hsv_s=0.0,        
        hsv_v=0.0,    
        batch=batch,
        workers=workers,
        imgsz=imgsz
    )

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for ID card detection.")
    parser.add_argument("--data_path", default="data/data.yaml", help="Path to data.yaml configuration file")
    parser.add_argument("--model_name", default="yolov8n.pt", help="Pre-trained YOLOv8 model (e.g., yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for training")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size for training")
    args = parser.parse_args()

    train_yolov8(args.data_path, args.model_name, args.epochs, args.batch, args.workers, args.imgsz)
