import os
import cv2
import argparse

def crop_images(image_folder, label_folder, crop_folder, confidence_threshold=0.7):
    # Create output directory if it doesn't exist
    os.makedirs(crop_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")

            if not os.path.exists(label_path):
                print(f"No label file for {filename}, skipping.")
                continue

            # Read image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Read bounding box labels
            with open(label_path, 'r') as file:
                lines = [line.strip() for line in file.readlines() if len(line.strip().split()) == 6]

            crop_count = 0
            crops_info = []

            # Process each bounding box
            for idx, line in enumerate(lines):
                class_id, x_c, y_c, w, h, conf = map(float, line.split())

                # Convert normalized coordinates to pixel values
                x_center = x_c * width
                y_center = y_c * height
                bbox_width = w * width
                bbox_height = h * height

                # Calculate bounding box corners
                x1 = int(max(0, x_center - bbox_width / 2))
                y1 = int(max(0, y_center - bbox_height / 2))
                x2 = int(min(width, x_center + bbox_width / 2))
                y2 = int(min(height, y_center + bbox_height / 2))

                # Crop image
                crop = image[y1:y2, x1:x2]
                crops_info.append((crop, conf, idx + 1))

            # Select crops based on confidence threshold
            selected_crops = []
            if len(crops_info) == 1:
                selected_crops = crops_info
            elif len(crops_info) == 2:
                c1, c2 = crops_info
                if c1[1] >= confidence_threshold and c2[1] >= confidence_threshold:
                    selected_crops = [c1, c2]
                elif c1[1] >= confidence_threshold or c2[1] >= confidence_threshold:
                    selected_crops = [c1 if c1[1] >= c2[1] else c2]
                else:
                    selected_crops = [c1 if c1[1] >= c2[1] else c2]
            else:
                print(f"Skipping {filename}: does not have 1 or 2 boxes.")
                continue

            # Save cropped images
            for crop, conf, box_idx in selected_crops:
                if len(selected_crops) == 1:
                    crop_filename = f"{os.path.splitext(filename)[0]}.jpg"
                else:
                    crop_filename = f"{os.path.splitext(filename)[0]}_conf{conf:.2f}_box{box_idx}.jpg"

                crop_path = os.path.join(crop_folder, crop_filename)
                cv2.imwrite(crop_path, crop)
                crop_count += 1

            if crop_count == 0:
                print(f"No crops saved from {filename}")
            else:
                print(f"Saved {crop_count} crop(s) from {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop ID cards from images using YOLOv8 bounding boxes.")
    parser.add_argument("--image_folder", default="data/test", help="Path to input images")
    parser.add_argument("--label_folder", default="data/predictions/labels", help="Path to YOLOv8 label files")
    parser.add_argument("--crop_folder", default="data/processed/cropped", help="Path to save cropped images")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold for bounding boxes")
    args = parser.parse_args()

    crop_images(args.image_folder, args.label_folder, args.crop_folder, args.confidence_threshold)
