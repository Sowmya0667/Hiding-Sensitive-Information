from ultralytics import YOLO
import argparse

def predict_yolov8(model_path, source, output_dir, conf=0.25, iou=0.7):
    """
    Predict bounding boxes for ID cards using a trained YOLOv8 model.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model weights.
        source (str): Path to the input images or directory.
        output_dir (str): Directory to save prediction results.
        conf (float): Confidence threshold for predictions.
        iou (float): IoU threshold for non-max suppression.
    """
    # Initialize YOLO model
    model = YOLO(model_path)

    # Perform prediction
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name="predict_train"
    )

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ID card bounding boxes using a trained YOLOv8 model.")
    parser.add_argument("--model_path", default="models/yolov8_id_card.pt", help="Path to trained YOLOv8 model weights")
    parser.add_argument("--source", default="data/test", help="Path to input images or directory")
    parser.add_argument("--output_dir", default="data/predictions", help="Directory to save prediction results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for non-max suppression")
    args = parser.parse_args()

    predict_yolov8(args.model_path, args.source, args.output_dir, args.conf, args.iou)
