import aspose.ocr as ocr
import cv2
import numpy as np
import os
import math
import argparse

def deskew_images(input_folder, output_folder):
    """
    Deskew images using Aspose OCR to calculate skew angle and OpenCV to rotate images.
    
    Args:
        input_folder (str): Path to the folder containing cropped images.
        output_folder (str): Path to save deskewed images.
    """
    # Initialize Aspose OCR
    api = ocr.AsposeOcr()

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process each image
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)

        # Load image and calculate skew
        ocr_input = ocr.OcrInput(ocr.InputType.SINGLE_IMAGE)
        ocr_input.add(img_path)
        angles = api.calculate_skew(ocr_input)

        for angle in angles:
            skew_angle = angle.angle
            print(f"File: {angle.source}")
            print(f"Skew angle: {skew_angle:.1f}°")

            # Load image using OpenCV
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                print(f"❌ Failed to load image: {img_path}")
                continue

            # Calculate new dimensions to prevent cropping
            (h, w) = img_cv.shape[:2]
            angle_rad = math.radians(abs(skew_angle))
            new_w = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))
            new_h = int(abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad)))

            center = (w // 2, h // 2)
            new_center = (new_w // 2, new_h // 2)

            # Create rotation matrix and adjust for new dimensions
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            M[0, 2] += new_center[0] - center[0]
            M[1, 2] += new_center[1] - center[1]

            # Rotate image
            rotated = cv2.warpAffine(img_cv, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Save deskewed image
            output_path = os.path.join(output_folder, f"{img_file}")
            cv2.imwrite(output_path, rotated)
            print(f"✅ Saved deskewed image to: {output_path}")

    print("🎉 Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deskew cropped images using Aspose OCR and OpenCV.")
    parser.add_argument("--input_folder", default="data/processed/cropped", help="Path to folder with cropped images")
    parser.add_argument("--output_folder", default="data/processed/aligned", help="Path to save deskewed images")
    args = parser.parse_args()

    deskew_images(args.input_folder, args.output_folder)
