from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2
import os
import json
import argparse

def extract_text(input_folder, output_image_folder, output_json_folder, font_path=None):
    """
    Extract text and coordinates from images using PaddleOCR, split text based on keywords or colons,
    and save annotated images and JSON files.

    Args:
        input_folder (str): Path to folder with aligned images.
        output_image_folder (str): Path to save annotated images.
        output_json_folder (str): Path to save JSON files with text and coordinates.
        font_path (str, optional): Path to font file for drawing OCR results.
    """
    # Initialize OCR
    ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False)

    # Create output directories
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_json_folder, exist_ok=True)

    # List of keywords to split after
    keywords = [
        "Name", "Husband", "Husband's Name", "Mother's Name", "DOB", "Date of Birth",
        "Date of Birth/Age", "Address", "YoB", "Year of Birth", "S/W/D", "Licence No",
        "Number", "Son/Daughter/Wife of", "DL NO", "Father", "Son of", "No.", "DL",
        "Driving Licence No", "S/W/D of", "Father's Name", "Permanent Account Number",
        "Passport No.", "Surname", "Given Name(s)", "Place of Birth", "Elector's Name",
        "Relation's Name", "Mobile No.", "Phone Number", "Phone No.", "Mobile Number",
        "Bill No.", "Zipcode"
    ]

    def split_box(box, split_ratio):
        """
        Split a quadrilateral box horizontally at split_ratio (between 0 and 1).
        """
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        x_left = x1 + (x2 - x1) * split_ratio
        x_right = x4 + (x3 - x4) * split_ratio

        box1 = [[x1, y1], [x_left, y2], [x_right, y3], [x4, y4]]
        box2 = [[x_left, y2], [x2, y2], [x3, y3], [x_right, y3]]
        return [box1, box2]

    def find_keyword_split(text):
        """
        Check if text contains any keyword and return the keyword and split position.
        Returns (keyword, split_index) if found, else (None, None).
        """
        text_lower = text.lower()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                start_idx = text_lower.find(keyword_lower)
                end_idx = start_idx + len(keyword)
                if end_idx < len(text) and text[end_idx] == ' ':
                    end_idx += 1
                return keyword, end_idx
        return None, None

    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            print(f"Processing {filename} ...")

            # Perform OCR
            result = ocr.ocr(img_path)
            inner_result = result[0] if result else []

            boxes = []
            texts = []
            json_data = []

            for res in inner_result:
                box, (text, _) = res
                # Try to split based on keywords
                keyword, split_idx = find_keyword_split(text)
                if keyword and split_idx < len(text):
                    # Split text after the keyword
                    part1 = text[:split_idx].strip()
                    part2 = text[split_idx:].strip()
                    if len(part1) > 0 and len(part2) > 0:
                        total_len = len(part1) + len(part2)
                        ratio = len(part1) / total_len
                        split_boxes = split_box(box, ratio)
                        for part_text, part_box in zip([part1, part2], split_boxes):
                            coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in part_box]
                            json_data.append({"text": part_text, "coordinates": coordinates})
                            boxes.append(part_box)
                            texts.append(part_text)
                    else:
                        # Fallback if splitting results in empty parts
                        coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                        json_data.append({"text": text, "coordinates": coordinates})
                        boxes.append(box)
                        texts.append(text)
                elif ':' in text:
                    # Try splitting on ':'
                    parts = text.split(':', 1)
                    parts = [part.strip() for part in parts]
                    if len(parts[0]) > 0 and len(parts[1]) > 0:
                        total_len = len(parts[0]) + len(parts[1])
                        ratio = len(parts[0]) / total_len
                        split_boxes = split_box(box, ratio)
                        for part_text, part_box in zip(parts, split_boxes):
                            coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in part_box]
                            json_data.append({"text": part_text, "coordinates": coordinates})
                            boxes.append(part_box)
                            texts.append(part_text)
                    else:
                        # Fallback if splitting on ':' fails
                        coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                        json_data.append({"text": text, "coordinates": coordinates})
                        boxes.append(box)
                        texts.append(text)
                else:
                    # No keyword or ':', use the full text and box
                    coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                    json_data.append({"text": text, "coordinates": coordinates})
                    boxes.append(box)
                    texts.append(text)

            # Draw annotated image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = draw_ocr(img, boxes, texts, None, font_path=font_path)
            annotated_img = Image.fromarray(annotated)

            # Save annotated image
            annotated_image_path = os.path.join(output_image_folder, filename)
            annotated_img.save(annotated_image_path)

            # Save JSON
            base_name = os.path.splitext(filename)[0]
            json_output_path = os.path.join(output_json_folder, f"{base_name}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            print(f"Saved annotated image and JSON for {filename}\n")

    print("All images processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text and coordinates from images using PaddleOCR.")
    parser.add_argument("--input_folder", default="data/processed/aligned", help="Path to folder with aligned images")
    parser.add_argument("--output_image_folder", default="data/processed/paddle_result/images", help="Path to save annotated images")
    parser.add_argument("--output_json_folder", default="data/processed/paddle_result/annotations", help="Path to save JSON files")
    parser.add_argument("--font_path", default=None, help="Path to font file for drawing OCR results (optional)")
    args = parser.parse_args()

    extract_text(args.input_folder, args.output_image_folder, args.output_json_folder, args.font_path)
