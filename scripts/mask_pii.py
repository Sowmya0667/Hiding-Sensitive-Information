import os
import json
from PIL import Image, ImageDraw
import glob
import shutil
import argparse
from typing import Dict, List

def mask_pii(json_folder: str, image_folder: str, output_folder: str) -> None:
    """
    Mask PII coordinates from JSON annotations on aligned images using black rectangles.

    Args:
        json_folder (str): Path to folder with PII JSON annotations.
        image_folder (str): Path to folder with aligned images.
        output_folder (str): Path to save masked images.
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]

        # Get list of JSON files
        json_files = glob.glob(os.path.join(json_folder, "*.json"))
        json_files.sort()  # Sort for consistent processing order

        print(f"Found {len(json_files)} JSON files to process.")

        # Process all JSON files
        for idx, json_file in enumerate(json_files, 1):
            json_filename = os.path.splitext(os.path.basename(json_file))[0]
            print(f"\nProcessing file {idx}/{len(json_files)}: {json_filename}")

            # Find corresponding image file
            image_path = None
            for ext in image_extensions:
                potential_paths = glob.glob(os.path.join(image_folder, f"{json_filename}{ext}"))
                if potential_paths:
                    image_path = potential_paths[0]
                    break

            if not image_path:
                print(f"No image found for {json_filename}. Skipping...")
                continue

            # Load JSON data
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {json_filename}: {e}. Skipping...")
                continue

            # Load the image
            try:
                image = Image.open(image_path).convert("RGB")
                masked_image = image.copy()
                draw = ImageDraw.Draw(masked_image)
            except Exception as e:
                print(f"Error opening image {image_path}: {e}. Skipping...")
                continue

            # Extract PII fields with coordinates
            coordinates_found = False
            for field, info in data.items():
                if isinstance(info, dict) and info.get("value") != "NONE" and info.get("coordinates"):
                    for coord_set in info["coordinates"]:
                        if coord_set:  # Ensure coordinate set is not empty
                            try:
                                x_coords = [point["x"] for point in coord_set]
                                y_coords = [point["y"] for point in coord_set]
                                left = max(0, min(x_coords))
                                top = max(0, min(y_coords))
                                right = min(image.width, max(x_coords))
                                bottom = min(image.height, max(y_coords))

                                if right > left and bottom > top:
                                    # Draw a black rectangle with thickness 3
                                    for offset in range(3):
                                        draw.rectangle(
                                            [left - offset, top - offset, right + offset, bottom + offset],
                                            outline="black",
                                            width=1
                                        )
                                    draw.rectangle([left, top, right, bottom], fill="black")
                                    coordinates_found = True
                                    print(f"Masked PII field '{field}' at coordinates: {coord_set}")
                            except KeyError as e:
                                print(f"Invalid coordinate format in field {field}: {e}. Skipping...")
                                continue

            # Define output path
            output_path = os.path.join(output_folder, f"{json_filename}.jpg")

            # If no coordinates were found, copy the original image
            if not coordinates_found:
                print(f"No PII coordinates found in {json_filename}. Copying original image...")
                shutil.copy(image_path, output_path)
                print(f"Copied image to: {output_path}")
                continue

            # Save the masked image
            try:
                masked_image.save(output_path, quality=95)
                print(f"Saved masked image: {output_path}")
            except Exception as e:
                print(f"Error saving image {output_path}: {e}. Skipping...")
                continue

        # Process unmatched images (images without corresponding JSON files)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.sort()

        for image_path in image_files:
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(json_folder, f"{image_filename}.json")
            if not os.path.exists(json_path):
                output_path = os.path.join(output_folder, f"{image_filename}.jpg")
                shutil.copy(image_path, output_path)
                print(f"Copied unmatched image: {output_path}")

        print("Processing complete.")

    except Exception as e:
        print(f"Error in mask_pii: {e}")

def main():
    parser = argparse.ArgumentParser(description="Mask PII coordinates from JSON annotations on aligned images.")
    parser.add_argument("--json_folder", default="data/processed/pii", help="Path to folder with PII JSON annotations")
    parser.add_argument("--image_folder", default="data/processed/aligned", help="Path to folder with aligned images")
    parser.add_argument("--output_folder", default="data/processed/masked", help="Path to save masked images")
    args = parser.parse_args()

    mask_pii(args.json_folder, args.image_folder, args.output_folder)

if __name__ == "__main__":
    main()
