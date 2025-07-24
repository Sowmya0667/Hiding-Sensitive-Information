import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import zipfile
import os
import glob
import shutil
import base64
import requests
import re
import time
import hashlib
import random
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import aspose.ocr
import math
from paddleocr import PaddleOCR, draw_ocr
import pdf2image
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid

class Config:
    """Configuration class for storing constants and settings."""
    TEMP_DIR = os.path.join(os.path.dirname(__file__), "pii_masking_temp")
    GOOGLE_DRIVE_FILE_ID = "1BUHPtTeZp1vDTfuXX86qPtceYxNxdbp4"
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    API_CALLS_PER_MINUTE = 4
    SECONDS_PER_MINUTE = 60
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"

    @staticmethod
    def setup_temp_dir():
        """Create the temporary directory if it doesn't exist."""
        os.makedirs(Config.TEMP_DIR, exist_ok=True)

    @staticmethod
    def validate_config():
        """Validate that required configuration settings are present."""
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in a .env file.")
        if not os.path.exists(Config.FONT_PATH):
            raise ValueError(f"Font file not found at {Config.FONT_PATH}. Please ensure the Arial font is installed at C:/Windows/Fonts/arial.ttf.")

class ModelLoader:
    """Class to handle downloading and loading of YOLO and OCR models."""
    @staticmethod
    @st.cache_resource(show_spinner="Downloading model weights...")
    def download_weights_from_drive(file_id: str, destination: str) -> str:
        """Download the weights file from Google Drive with retry logic."""
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        session = requests.Session()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = session.get(Config.DOWNLOAD_URL, stream=True, timeout=30)
                token = get_confirm_token(response)
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(Config.DOWNLOAD_URL, params=params, stream=True, timeout=30)
                response.raise_for_status()
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                if not os.path.exists(destination) or os.path.getsize(destination) == 0:
                    raise ValueError("Downloaded file is empty or missing")
                return destination
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    st.error(f"Failed to download weights file after {max_retries} attempts: {str(e)}")
                    st.stop()

    @staticmethod
    @st.cache_resource(show_spinner="Loading models...")
    def load_models() -> Tuple[YOLO, PaddleOCR]:
        """Load YOLO and PaddleOCR models."""
        Config.validate_config()  # Validate config before loading models
        weights_path = os.path.join(Config.TEMP_DIR, "best.pt")
        if not os.path.exists(weights_path):
            ModelLoader.download_weights_from_drive(Config.GOOGLE_DRIVE_FILE_ID, weights_path)
        try:
            yolo_model = YOLO(weights_path)
            return yolo_model, PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False)
        except Exception as e:
            st.error(f"Failed to load YOLO model from {weights_path}: {str(e)}")
            st.cache_resource.clear()
            st.experimental_rerun()

class HelperFunctions:
    """Utility functions for PII processing."""
    @staticmethod
    def encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            st.error(f"Failed to encode image {image_path}: {str(e)}")
            return None

    @staticmethod
    def is_english_text(text: str) -> bool:
        return bool(re.match(r'^[a-zA-Z0-9\s.,-/:&]*$', text))

    @staticmethod
    def is_valid_name(text: str) -> bool:
        if not text or len(text) < 3:
            return False
        return bool(re.match(r'^[a-zA-Z\s.&]+$', text))

    @staticmethod
    def split_box(box: List[List[float]], split_ratio: float) -> List[List[List[float]]]:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        x_left = x1 + (x2 - x1) * split_ratio
        x_right = x4 + (x3 - x4) * split_ratio
        box1 = [[x1, y1], [x_left, y2], [x_right, y3], [x4, y4]]
        box2 = [[x_left, y2], [x2, y2], [x3, y3], [x_right, y3]]
        return [box1, box2]

    @staticmethod
    def find_keyword_split(text: str) -> Tuple[Optional[str], Optional[int]]:
        text_lower = text.lower()
        for keyword in Config.KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                start_idx = text_lower.find(keyword_lower)
                end_idx = start_idx + len(keyword)
                if end_idx < len(text) and text[end_idx] == ' ':
                    end_idx += 1
                return keyword, end_idx
        return None, None

    @staticmethod
    def replace_null_with_none(data: Dict) -> Dict:
        if isinstance(data, dict):
            return {k: HelperFunctions.replace_null_with_none(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [HelperFunctions.replace_null_with_none(item) for item in data]
        elif data is None:
            return "NONE"
        return data

    @staticmethod
    def is_valid_coordinate_format(coordinates: List) -> bool:
        try:
            if not isinstance(coordinates, list) or not coordinates:
                return False
            for coord_set in coordinates:
                if isinstance(coord_set, dict):
                    keys = list(coord_set.keys())
                    x_count = keys.count('x')
                    y_count = keys.count('y')
                    if x_count != y_count or x_count < 1:
                        return False
                    key_sequence = [key for key in keys if key in ['x', 'y']]
                    for i in range(0, len(key_sequence) - 1, 2):
                        if key_sequence[i] != 'x' or key_sequence[i + 1] != 'y':
                            return False
                    for key, value in coord_set.items():
                        if not isinstance(value, (int, float)):
                            return False
                elif isinstance(coord_set, list):
                    if not coord_set or len(coord_set) > 4:
                        return False
                    for point in coord_set:
                        if not isinstance(point, dict) or sorted(point.keys()) != ['x', 'y']:
                            return False
                        if not all(isinstance(point[k], (int, float)) for k in point):
                            return False
                else:
                    return False
            return True
        except Exception as e:
            print(f"ERROR: Exception in is_valid_coordinate_format: {str(e)}")
            return False

    @staticmethod
    def coordinate_hook(pairs):
        if len(pairs) > 0 and all(key in ['x', 'y'] for key, value in pairs):
            x_count = sum(1 for key, _ in pairs if key == 'x')
            y_count = sum(1 for key, _ in pairs if key == 'y')
            if x_count > 1 or y_count > 1:
                return HelperFunctions.merge_duplicate_keys(pairs)
            else:
                return dict(pairs)
        else:
            result = {}
            for key, value in pairs:
                if key in result:
                    if isinstance(result[key], list):
                        result[key].append(value)
                    else:
                        result[key] = [result[key], value]
                else:
                    result[key] = value
            return result

    @staticmethod
    def merge_duplicate_keys(pairs):
        x_vals = [value for key, value in pairs if key == 'x']
        y_vals = [value for key, value in pairs if key == 'y']
        points = [{'x': x_vals[i], 'y': y_vals[i]} for i in range(min(len(x_vals), y_vals))]
        return points if points else []

    @staticmethod
    def fix_coordinates_format(coordinates: list) -> list:
        try:
            if not coordinates or not isinstance(coordinates, list):
                return []
            flat_points = []
            if all(isinstance(coord_set, list) and 1 <= len(coord_set) <= 4 and 
                   all(isinstance(point, dict) and sorted(point.keys()) == ['x', 'y'] and 
                       all(isinstance(point[k], (int, float)) for k in point) for point in coord_set) 
                   for coord_set in coordinates):
                flat_points = [point for coord_set in coordinates for point in coord_set]
            if not flat_points and all(isinstance(coord, dict) and sorted(coord.keys()) == ['x', 'y'] and 
                                      all(isinstance(coord[k], (int, float)) for k in coord) for coord in coordinates):
                flat_points = coordinates
            if flat_points:
                grouped_coordinates = [flat_points[i:i+4] for i in range(0, len(flat_points), 4)]
                return grouped_coordinates
            all_bounding_boxes = []
            for coord_set in coordinates:
                if isinstance(coord_set, dict):
                    keys = list(coord_set.keys())
                    x_count = keys.count('x')
                    y_count = keys.count('y')
                    if x_count >= 1 and x_count == y_count:
                        points = HelperFunctions.merge_duplicate_keys(list(coord_set.items()))
                        if points and len(points) <= 4:
                            all_bounding_boxes.append(points)
                    elif x_count == 1 and y_count == 1 and sorted(keys) == ['x', 'y']:
                        if all(isinstance(coord_set[k], (int, float)) for k in ['x', 'y']):
                            all_bounding_boxes.append([coord_set])
                elif isinstance(coord_set, list):
                    points = []
                    for item in coord_set:
                        if isinstance(item, dict) and sorted(item.keys()) == ['x', 'y'] and all(isinstance(item[k], (int, float)) for k in item):
                            points.append(item)
                    if points:
                        for i in range(0, len(points), 4):
                            all_bounding_boxes.append(points[i:i+4])
            return all_bounding_boxes
        except Exception as e:
            print(f"ERROR: Exception in fix_coordinates_format: {str(e)}")
            return []

class PIIExtractor:
    """Class to handle PII extraction from documents."""
    @staticmethod
    def extract_passport_mrz(json_data: List[Dict]) -> Dict[str, Optional[Dict]]:
        try:
            result = {"Surname": None, "Given Name(s)": None}
            mrz_line = None
            full_text = " ".join([block["text"] for block in json_data])
            full_text = re.sub(r'[^A-Z0-9<]', '', full_text)
            full_text = re.sub(r'<{2,}', '<<', full_text)
            mrz_patterns = [
                r'P<IND[A-Z<]+<<[A-Z<]+',
                r'P<IND[A-Z<]+<<[A-Z<]+<<[A-Z<]+',
                r'P<IND[A-Z<]+<<[A-Z<]+<<[A-Z<]+<<[A-Z<]+',
                r'P<IND[A-Z<]+<<[A-Z<]+<<[A-Z<]+<<[A-Z<]+<<[A-Z<]+'
            ]
            for pattern in mrz_patterns:
                mrz_match = re.search(pattern, full_text)
                if mrz_match:
                    start_idx = mrz_match.start()
                    mrz_line = full_text[start_idx:start_idx + 44]
                    break
            if not mrz_line:
                for block in json_data:
                    text = block["text"]
                    text = re.sub(r'[^A-Z0-9<]', '', text)
                    text = re.sub(r'<{2,}', '<<', text)
                    for pattern in mrz_patterns:
                        if re.match(pattern, text):
                            mrz_line = text[:44]
                            break
                    if mrz_line:
                        break
            if not mrz_line:
                for block in json_data:
                    text = block["text"].upper()
                    surname_match = re.search(r'^(?:SURNAME\s*/)?([A-Z\s]+?)(?=\s+[A-Z\s]+\d{7}\b|$)', text, re.IGNORECASE)
                    if surname_match:
                        surname = surname_match.group(1).strip()
                        if HelperFunctions.is_valid_name(surname):
                            result["Surname"] = {"value": surname, "coordinates": block.get("coordinates", [])}
                    given_name_match = re.search(r'(?:GIVEN NAME\s*[/:])?\s*([A-Z\s]+?)(?=\s+INDIAN|\s+M\s+\d{2}/\d{2}/\d{4}\b|$)', text, re.IGNORECASE)
                    if given_name_match:
                        given_names = given_name_match.group(1).strip()
                        if HelperFunctions.is_valid_name(given_names):
                            result["Given Name(s)"] = {"value": given_names, "coordinates": block.get("coordinates", [])}
                return result
            name_part = mrz_line[5:]
            name_parts = name_part.split('<<')
            if len(name_parts) < 2:
                return result
            surname = name_parts[0].replace('IND', '', 1).replace('<', ' ').strip()
            given_names = ' '.join([part.replace('<', ' ').strip() for part in name_parts[1:] if part.strip()])
            if not HelperFunctions.is_valid_name(surname) or not HelperFunctions.is_valid_name(given_names):
                return result
            surname_coords = []
            given_names_coords = []
            for block in json_data:
                block_text = block.get("text", "").upper()
                if surname in block_text:
                    surname_coords.append(block.get("coordinates", []))
                if any(name in block_text for name in given_names.split()):
                    given_names_coords.append(block.get("coordinates", []))
            result["Surname"] = {"value": surname, "coordinates": surname_coords}
            result["Given Name(s)"] = {"value": given_names, "coordinates": given_names_coords}
            return result
        except Exception as e:
            print(f"Error extracting passport MRZ: {str(e)}")
            return {"Surname": None, "Given Name(s)": None}

    @staticmethod
    def extract_aadhaar_id(json_data: List[Dict], text: str, phone_number: str = "NONE") -> Dict[str, Optional[str]]:
        try:
            aadhaar_id = None
            coordinates = []
            for block in json_data:
                block_text = block.get("text", "")
                block_text_cleaned = re.sub(r'[^0-9]', '', block_text)
                if re.match(r'[2-9]\d{11}$', block_text_cleaned):
                    aadhaar_id = block_text_cleaned
                    coords = block.get("coordinates", [])
                    if coords and HelperFunctions.is_valid_coordinate_format([coords]):
                        coordinates.append(coords)
                    break
            if not aadhaar_id:
                cleaned_text = re.sub(r'[^0-9]', '', text)
                aadhaar_match = re.search(r'[2-9]\d{11}', cleaned_text)
                if aadhaar_match:
                    aadhaar_id = aadhaar_match.group(0)
                    aadhaar_variations = [
                        aadhaar_id,
                        re.sub(r'(\d{4})(\d{4})(\d{4})', r'\1 \2 \3', aadhaar_id),
                        aadhaar_id[:8] + " " + aadhaar_id[8:],
                        aadhaar_id[:4] + " " + aadhaar_id[4:],
                        re.sub(r'(\d{4})(\d{4})(\d{4})', r'\1-\2-\3', aadhaar_id),
                        aadhaar_id[:8] + "-" + aadhaar_id[8:],
                        aadhaar_id[:4] + "-" + aadhaar_id[4:]
                    ]
                    for block in json_data:
                        block_text = block.get("text", "")
                        block_text_cleaned = re.sub(r'[^0-9]', '', block_text)
                        if any(variation in block_text or aadhaar_id in block_text_cleaned for variation in aadhaar_variations):
                            coords = block.get("coordinates", [])
                            if coords and HelperFunctions.is_valid_coordinate_format([coords]):
                                coordinates.append(coords)
                else:
                    return {"value": "NONE", "coordinates": []}
            if not aadhaar_id:
                return {"value": "NONE", "coordinates": []}
            if not coordinates:
                return {"value": aadhaar_id, "coordinates": []}
            return {"value": aadhaar_id, "coordinates": HelperFunctions.fix_coordinates_format(coordinates)}
        except Exception as e:
            print(f"Error extracting Aadhaar ID: {str(e)}")
            return {"value": "NONE", "coordinates": []}

    @staticmethod
    def extract_pii_with_gemini(text: str, json_data: List[Dict], image_path: Optional[str] = None, max_retries: int = 3) -> Dict:
        try:
            Config.validate_config()  # Ensure GEMINI_API_KEY is set
            GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={Config.GEMINI_API_KEY}"
            text_safe = text.replace('{', '{{').replace('}', '}}')
            json_data_str = json.dumps(json_data, ensure_ascii=False).replace('{', '{{').replace('}', '}}')
            prompt = (
                    "You are tasked with extracting Personally Identifiable Information (PII) from documents, including mixed content such as text reports and embedded ID cards (e.g., Aadhaar Card, PAN Card, Passport, Driving License, Voter ID Card). Use the provided text and JSON annotations, with an optional image to improve text quality and PII detection. If an image is provided, use it only to enhance text quality and assist in identifying PII fields, not for extracting text coordinates. Coordinates for PII fields must be sourced exclusively from the provided JSON annotations. Do not extract or adjust coordinates based on the image.\n\n"
                    "The document may contain a mix of free text (e.g., reports, acknowledgments) and embedded ID cards. Identify the document type as 'Mixed' if both text and an ID card are present, or classify the specific ID card type (e.g., 'PAN Card') if an ID card is dominant, or 'Text Only' if no ID card is detected. Extract PII from both the free text and any embedded ID card.\n\n"
                    "**Extracting All Names**: Identify all distinct names in the document that are not labeled as 'Father's Name', 'Mother's Name', or 'Husband's Name'. The primary name (e.g., the report author or main individual) should be assigned to the 'Name' field. For additional names (e.g., mentors, teammates), create fields named 'Name1', 'Name2', etc., in the order they appear. A name is valid if it contains only alphabetic characters, spaces, or '&', and is at least 3 characters long. Include coordinates from all JSON annotation text blocks that contain the name or any part of it, using case-insensitive and space-insensitive matching.\n\n"
                    "If an image is provided, use it only to improve text quality to correct OCR errors and identify PII fields, not for extracting text coordinates. "
                    "Coordinates for PII fields must be sourced exclusively from the provided JSON annotations. Do not extract or adjust coordinates based on the image. The output must preserve the format of the coordinates as a list of dictionaries, each containing 'x' and 'y' keys, exactly as provided in the annotation. For example, if the annotation is: "
                    "{\n"
                    "  \"text\": \"VUAY KUMAR\",\n"
                    "  \"coordinates\": [\n"
                    "    {\"x\": 29, \"y\": 283},\n"
                    "    {\"x\": 123, \"y\": 285},\n"
                    "    {\"x\": 122, \"y\": 303},\n"
                    "    {\"x\": 29, \"y\": 301}\n"
                    "  ]\n"
                    "}\n"
                    "Then the output must be:\n"
                    "\"Name\": {\n"
                    "  \"value\": \"VIJAY KUMAR\",\n"
                    "  \"coordinates\": [\n"
                    "    {\"x\": 29, \"y\": 283},\n"
                    "    {\"x\": 123, \"y\": 285},\n"
                    "    {\"x\": 122, \"y\": 303},\n"
                    "    {\"x\": 29, \"y\": 301}\n"
                    "  ]\n"
                    "}\n"
                    "Ensure coordinates are copied verbatim from the annotation, with no alterations to values or structure. For multiple coordinates like:\n"
                    "{\n"
                    "  \"text\": \"SUMRA\",\n"
                    "  \"coordinates\": [\n"
                    "    {\"x\": 292, \"y\": 140},\n"
                    "    {\"x\": 360, \"y\": 146},\n"
                    "    {\"x\": 358, \"y\": 166},\n"
                    "    {\"x\": 290, \"y\": 160}\n"
                    "  ]\n"
                    "}\n"
                    "and\n"
                    "{\n"
                    "  \"text\": \"SAJID UMAR\",\n"
                    "  \"coordinates\": [\n"
                    "    {\"x\": 290, \"y\": 191},\n"
                    "    {\"x\": 421, \"y\": 199},\n"
                    "    {\"x\": 420, \"y\": 220},\n"
                    "    {\"x\": 289, \"y\": 211}\n"
                    "  ]\n"
                    "}\n"
                    "Then the output should be:\n"
                    "\"Name\": {\n"
                    "  \"value\": \"SUMRA SAJID UMAR\",\n"
                    "  \"coordinates\": [\n"
                    "    {\"x\": 292, \"y\": 140},\n"
                    "    {\"x\": 360, \"y\": 146},\n"
                    "    {\"x\": 358, \"y\": 166},\n"
                    "    {\"x\": 290, \"y\": 160},\n"
                    "    {\"x\": 290, \"y\": 191},\n"
                    "    {\"x\": 421, \"y\": 199},\n"
                    "    {\"x\": 420, \"y\": 220},\n"
                    "    {\"x\": 289, \"y\": 211}\n"
                    "  ]\n"
                    "}\n"
                    "If a valid PII field is detected but its text is not found in any annotation block, set 'coordinates': [] while still returning the 'value'."
                    "Use only English text; ignore non-English text (e.g., extract 'Mehboob Rajput' from 'HI Mehboob Rajput'). "
                    "**Extracting All Names**: Identify all distinct names in the document that are not labeled as Father's Name, Mother's Name, or Husband's Name. The primary name (e.g., the author or main individual) should be assigned to the 'Name' field. For additional names (e.g., mentors, teammates), create fields named 'Name1', 'Name2', etc., in the order they appear. A name is valid if it contains only alphabetic characters, spaces, or '&' (for water bills), and is at least 3 characters long. For each name field, include coordinates from all JSON annotation text blocks that contain the name or any part of it (e.g., for 'Mr. Amit Kumar', include blocks with 'Mr. Amit Kumar', 'Mr. Amit', or 'Kumar'), using case-insensitive and space-insensitive matching. "
                    "For each PII value, include all distinct coordinates from JSON annotations for every text block containing the PII value or its parts. "
                    "Do not extract coordinates from the image. The image is provided to enhance text quality and PII detection. "
                    "Return null for unclear or missing fields. Output as JSON.\n\n"
                    "- **Document Identification**: Identify the document type (Aadhaar Card, PAN Card, Passport, Driving License, Voter ID Card, Water Bill, or Unknown for non-ID documents like reports). Use text patterns, labels (e.g., 'GOVERNMENT OF INDIA', 'ELECTION COMMISSION'), or image content (if provided). "
                    "- For each document type, extract only its corresponding ID and set all other ID fields to {'value': 'NONE', 'coordinates': []}. For Aadhaar Card, extract only the Aadhaar ID (exactly 12 digits, e.g., '986709620066'), which may appear with spaces or without. For Voter ID Card, extract only the Voter ID (10-character alphanumeric, e.g., 'ABC1234567'). For PAN Card, extract only the PAN ID (10-character alphanumeric, e.g., 'ABCDE1234F'). For Passport, extract only the Passport ID (8-character alphanumeric, e.g., 'A1234567'). For Driving License, extract only the Driving License ID (alphanumeric, e.g., 'DL1234567890123'). For Water Bill, extract only the Bill Number (12-digit, e.g., '788659363282'). For unknown document types (e.g., reports, letters), set Document Type to 'Unknown' and extract names and other PII if present, setting ID fields to {'value': 'NONE', 'coordinates': []} unless detected. Phone Number: 10-digit starting with 6-9, distinct from Aadhaar ID or other IDs. "
                    "- Aadhaar Card (front and back) should be detected as 'Aadhaar Card', Voter ID Card as 'Voter ID Card'. Non-ID documents (e.g., reports, letters) should be 'Unknown'. "
                    "- **General**: The primary 'Name' is the main individual's name (e.g., report author). Additional names (e.g., mentors, teammates) should be assigned to 'Name1', 'Name2', etc. Names may include '&' for water bills. Ignore short fragments like 'Na'. "
                    "- **Aadhaar Card**: Extract Aadhaar ID (exactly 12-digit number, e.g., '986709620066'). Do not extract VID (16 digits). Use the image to verify and improve text accuracy if OCR output is unclear. Extract only the 12-digit Aadhaar number (e.g., '986709620066') that is clearly labeled as Aadhaar Number or appears in expected positions.Do NOT extract the following : a) Any 16-digit number (e.g., VID) b) Any 12-digit sequence that is part of a 16-digit number c) Any number unless you are confident it is a standalone 12-digit Aadhaar number based on both the image layout and text annotation"
                    "- **Phone Number**: Exactly 10-digit number starting with 6-9. A 12-digit number is Aadhaar ID, not Phone Number. "
                    "- **Voter ID Card**: Extract Voter ID (10-char code, e.g., 'RXH0704411'). Don't extract VID number as Voter ID "
                    "- **Passport**: Extract Passport ID (8-char, e.g., 'H1591111'). Use 'Place of Birth' for Address. Combine 'Surname' and 'Given Name(s)' into 'Name'. Prioritize MRZ data for Name, Date of Birth, and Passport ID if available. Extract Name from the first MRZ line, Passport ID (positions 1-8), and Date of Birth (positions 14-19, YYMMDD to DD/MM/YYYY) from the second MRZ line. Include coordinates from JSON annotations for MRZ text blocks. "
                    "- **Driving License**: Prioritize 'Date of Birth' over 'Issue Date'. Date format: DD/MM/YYYY. Father's Name may appear as 'Son/Daughter/Wife of'. "
                    "- **Water Bill**: Extract Bill Number (12-digit, e.g., '788659363282'). For names, include multiple individuals (e.g., 'Amit Kumar & Sunita Devi') in separate fields ('Name', 'Name1', etc.). "
                    "- **PAN Card**: Extract PAN ID (10-char, e.g., 'ABCDE1234F'). Name and Father's Name may be unlabeled; assume first two all-caps names before PAN ID or DOB. "
                    "- **Non-ID Documents (e.g., Reports)**: Extract names (e.g., author, mentors, teammates) as 'Name', 'Name1', 'Name2', etc. Set ID fields (Aadhaar ID, PAN ID, etc.) to {'value': 'NONE', 'coordinates': []} unless explicitly detected. "
                    "- Coordinates must be sourced exclusively from JSON annotations. If coordinates are invalid, return an empty list for that field. "
                    f"**Text**: {text_safe}\n\n"
                    f"**JSON Annotations**: {json_data_str}\n\n"
                    "**General Rules**:\n"
                    "- Use only English text; ignore non-English text.\n"
                    "- Use the image (if provided) to enhance text quality and PII detection, not for coordinates. "
                    "- Extract PII fields (e.g., Name, Name1, Name2, Date of Birth, Aadhaar ID, etc.) from text and JSON annotations. "
                    "- For each PII value, include all distinct coordinates from JSON annotations for every text block containing the value or its parts.\n"
                    "- If no ID card is detected but PII (e.g., names, dates) is present in text, set document type to 'Text Only'.\n"
                    "- If an ID card is detected alongside text, set document type to 'Mixed' and extract PII from both.\n"
                    "- Return JSON with the document type and PII fields, including dynamic 'NameX' fields:\n"
                    "- Return null or 'NONE' for unclear or missing fields.\n"
                    "- Output as JSON with the document type and PII fields, including dynamic 'NameX' fields based on the number of detected names:\n"

                    "```json\n"
                    "{\n"
                    "    \"Document Type\": \"Mixed|Text Only|Aadhaar Card|PAN Card|Passport|Driving License|Voter ID Card\",\n"
                    "    \"Name\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name1\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name2\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name3\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name4\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name5\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name6\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name7\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name8\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name9\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Name10\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Father's Name\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Mother's Name\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Husband's Name\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Phone Number\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Date of Birth\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Aadhaar ID\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"PAN ID\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Passport ID\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Driving License ID\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Voter ID\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"Address\": {\"value\": null, \"coordinates\": []},\n"
                    "    \"ZIP Code\": {\"value\": null, \"coordinates\": []}\n"
                    "}\n"
                    "```\n"
                    "- Include only the 'NameX' fields that correspond to detected names (e.g., if 5 names are found, include 'Name' through 'Name4', leaving higher indices as null or omitted). Ensure names are distinct and not duplicates of 'Father's Name', 'Mother's Name', or 'Husband's Name'.\n"
                    f"**Text**: {text_safe}\n\n"
                    f"**JSON Annotations**: {json_data_str}\n"
                ) 

            headers = {"Content-Type": "application/json"}
            parts = [{"text": prompt}]
            if image_path:
                print(f"Sending image {image_path} to Gemini API")
                encoded_image = HelperFunctions.encode_image(image_path)
                if encoded_image:
                    parts.append({"inlineData": {"mimeType": "image/jpeg", "data": encoded_image}})
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.5}
            }
            for attempt in range(max_retries):
                try:
                    response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    raw_response = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    if raw_response.startswith("```json") and raw_response.endswith("```"):
                        raw_response = raw_response[7:-3].strip()
                    parsed_result = json.loads(raw_response, object_pairs_hook=HelperFunctions.coordinate_hook)
                    if not isinstance(parsed_result, dict):
                        raise ValueError("Parsed result is not a dictionary")
                    for field in parsed_result:
                        if field != "Document Type" and isinstance(parsed_result[field], dict) and "coordinates" in parsed_result[field]:
                            raw_coords = parsed_result[field]["coordinates"]
                            if raw_coords:
                                try:
                                    parsed_result[field]["coordinates"] = HelperFunctions.fix_coordinates_format(raw_coords)
                                except Exception as e:
                                    print(f"Error fixing coordinates for field {field}: {str(e)}")
                                    parsed_result[field]["coordinates"] = []
                            else:
                                parsed_result[field]["coordinates"] = []
                    valid_fields = [
                        "Document Type", "Name", "Father's Name", "Mother's Name", "Husband's Name", "Phone Number",
                        "Date of Birth", "Aadhaar ID", "PAN ID", "Passport ID", "Driving License ID",
                        "Voter ID", "Address", "ZIP Code"
                    ]
                    parsed_result = {k: v for k, v in parsed_result.items() if k in valid_fields or (k.startswith("Name") and int(k.replace("Name", "")) <= 10)}
                    parsed_result = HelperFunctions.replace_null_with_none(parsed_result)
                    parsed_result['_raw_response'] = raw_response
                    return parsed_result
                except (requests.HTTPError, json.JSONDecodeError, ValueError) as e:
                    if isinstance(e, requests.HTTPError) and e.response.status_code == 429:
                        time.sleep(random.choice([90, 120]))
                        continue
                    print(f"Gemini API error: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    return PIIExtractor.default_pii_result()
            return PIIExtractor.default_pii_result()
        except Exception as e:
            print(f"Error in extract_pii_with_gemini: {str(e)}")
            return PIIExtractor.default_pii_result()

    @staticmethod
    def default_pii_result() -> Dict:
        """Return default PII result for error cases."""
        return {
            "Document Type": "Mixed",
            "Name": {"value": "NONE", "coordinates": []},
            "Name1": {"value": "NONE", "coordinates": []},
            "Name2": {"value": "NONE", "coordinates": []},
            "Name3": {"value": "NONE", "coordinates": []},
            "Name4": {"value": "NONE", "coordinates": []},
            "Name5": {"value": "NONE", "coordinates": []},
            "Name6": {"value": "NONE", "coordinates": []},
            "Name7": {"value": "NONE", "coordinates": []},
            "Name8": {"value": "NONE", "coordinates": []},
            "Name9": {"value": "NONE", "coordinates": []},
            "Name10": {"value": "NONE", "coordinates": []},
            "Father's Name": {"value": "NONE", "coordinates": []},
            "Mother's Name": {"value": "NONE", "coordinates": []},
            "Husband's Name": {"value": "NONE", "coordinates": []},
            "Phone Number": {"value": "NONE", "coordinates": []},
            "Date of Birth": {"value": "NONE", "coordinates": []},
            "Aadhaar ID": {"value": "NONE", "coordinates": []},
            "PAN ID": {"value": "NONE", "coordinates": []},
            "Passport ID": {"value": "NONE", "coordinates": []},
            "Driving License ID": {"value": "NONE", "coordinates": []},
            "Voter ID": {"value": "NONE", "coordinates": []},
            "Address": {"value": "NONE", "coordinates": []},
            "ZIP Code": {"value": "NONE", "coordinates": []},
            "_raw_response": None
        }

    @staticmethod
    def process_json_file(json_file_path: str, image_path: str = None, use_gemini: bool = True) -> Dict:
        try:
            print(f"Processing JSON file: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            text = " ".join([block['text'] for block in json_data])
            if use_gemini:
                result = PIIExtractor.extract_pii_with_gemini(text, json_data, image_path)
            else:
                result = PIIExtractor.default_pii_result()
                result["Document Type"] = "Unknown"
            result.pop('_raw_response', None)
            if result["Document Type"] == "Unknown":
                names = []
                for block in json_data:
                    text = block.get("text", "")
                    name_match = re.match(r'^(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*([A-Za-z\s]{3,})$', text.strip(), re.IGNORECASE)
                    if name_match:
                        name = name_match.group(1).strip()
                        if name not in [result.get(f, {}).get("value", "") for f in ["Name", "Father's Name", "Mother's Name", "Husband's Name"]] + [result.get(f"Name{i}", {}).get("value", "") for i in range(1, 10)]:
                            if name.lower() not in ['acknowledgement', 'abstract', 'introduction', 'university', 'coriolis', 'technologies']:
                                names.append((name, block.get("coordinates", [])))
                for i, (name, coords) in enumerate(names):
                    field = "Name" if i == 0 and result["Name"]["value"] == "NONE" else f"Name{i+1}"
                    result[field] = {"value": name, "coordinates": HelperFunctions.fix_coordinates_format([coords]) if coords else []}
            if result["Document Type"] == "Aadhaar Card" and result["Aadhaar ID"]["value"] == "NONE":
                aadhaar_result = PIIExtractor.extract_aadhaar_id(json_data, text, result["Phone Number"]["value"])
                result["Aadhaar ID"] = aadhaar_result
                if aadhaar_result["value"] != "NONE" and result["Phone Number"]["value"] == aadhaar_result["value"]:
                    result["Phone Number"] = {"value": "NONE", "coordinates": []}
            print(f"Gemini PII: {json.dumps({key: result[key].get('value') if key != 'Document Type' else result[key] for key in result}, indent=2, ensure_ascii=False)}")
            return result
        except Exception as e:
            print(f"Error in process_json_file: {str(e)}")
            return {
                "Document Type": "Unknown",
                "Name": {"value": "NONE", "coordinates": []},
                "Father's Name": {"value": "NONE", "coordinates": []},
                "Mother's Name": {"value": "NONE", "coordinates": []},
                "Husband's Name": {"value": "NONE", "coordinates": []},
                "Phone Number": {"value": "NONE", "coordinates": []},
                "Date of Birth": {"value": "NONE", "coordinates": []},
                "Aadhaar ID": {"value": "NONE", "coordinates": []},
                "PAN ID": {"value": "NONE", "coordinates": []},
                "Passport ID": {"value": "NONE", "coordinates": []},
                "Driving License ID": {"value": "NONE", "coordinates": []},
                "Voter ID": {"value": "NONE", "coordinates": []},
                "Address": {"value": "NONE", "coordinates": []},
                "ZIP Code": {"value": "NONE", "coordinates": []},
                "Bill Number": {"value": "NONE", "coordinates": []}
            }

class ImageProcessor:
    """Class to handle image processing, including conversion, detection, alignment, OCR, and PII masking."""
    @staticmethod
    def convert_to_jpg(image: Image.Image, output_path: str):
        image = image.convert("RGB")
        image.save(output_path, "JPEG", quality=95)

    @staticmethod
    def convert_pdf_to_images(pdf_path: str, output_dir: str) -> List[Tuple[str, int]]:
        try:
            images = pdf2image.convert_from_path(pdf_path)
            output_paths = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            for i, img in enumerate(images):
                output_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.jpg")
                ImageProcessor.convert_to_jpg(img, output_path)
                output_paths.append((output_path, i+1))
            return output_paths
        except Exception as e:
            st.error(f"Error converting PDF {pdf_path}: {str(e)}")
            return []

    @staticmethod
    def process_image(image_path: str, output_dir: str, yolo_model: YOLO, ocr_model: PaddleOCR, status_placeholder, original_file_name: str, page_number: Optional[int] = None) -> Tuple[Dict, List[Tuple[str, str]]]:
        status_placeholder.info("ID card detection has started...")
        results = yolo_model.predict(
            source=image_path,
            conf=0.25,
            iou=0.7,
            save=True,
            save_txt=True,
            save_conf=True,
            project=os.path.join(output_dir, "yolo_output"),
            name="predict"
        )
        status_placeholder.info("ID card detection has completed.")
        yolo_output_base = os.path.join(output_dir, "yolo_output")
        predict_dirs = sorted(glob.glob(os.path.join(yolo_output_base, "predict*")), key=os.path.getmtime, reverse=True)
        if not predict_dirs:
            status_placeholder.info("No ID card detected, using original image.")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cropped_images = [(image_path, base_name)]
        else:
            latest_predict_dir = predict_dirs[0]
            label_dir = os.path.join(latest_predict_dir, "labels")
            filename = os.path.basename(image_path)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(label_path):
                status_placeholder.info("No ID card detected, using original image.")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cropped_images = [(image_path, base_name)]
            else:
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                with open(label_path, 'r') as file:
                    lines = [line.strip() for line in file.readlines() if len(line.strip().split()) == 6]
                if not lines:
                    status_placeholder.info("No valid bounding boxes detected, using original image.")
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    cropped_images = [(image_path, base_name)]
                else:
                    crops_info = []
                    for idx, line in enumerate(lines):
                        class_id, x_c, y_c, w, h, conf = map(float, line.split())
                        if conf < 0.7:
                            continue
                        x_center = x_c * width
                        y_center = y_c * height
                        bbox_width = w * width
                        bbox_height = h * height
                        x1 = int(max(0, x_center - bbox_width / 2))
                        y1 = int(max(0, y_center - bbox_height / 2))
                        x2 = int(min(width, x_center + bbox_width / 2))
                        y2 = int(min(height, y_center + bbox_height / 2))
                        crop = image[y1:y2, x1:x2]
                        crops_info.append((crop, conf, idx + 1))
                    selected_crops = []
                    if len(crops_info) == 1:
                        selected_crops = crops_info
                    elif len(crops_info) == 2:
                        c1, c2 = crops_info
                        if c1[1] >= 0.7 and c2[1] >= 0.7:
                            selected_crops = [c1, c2]
                        elif c1[1] >= 0.7 or c2[1] >= 0.7:
                            selected_crops = [c1 if c1[1] >= c2[1] else c2]
                        else:
                            selected_crops = [c1 if c1[1] >= c2[1] else c2]
                    else:
                        status_placeholder.info(f"Skipping {filename}: does not have 1 or 2 boxes with confidence >= 0.7.")
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        cropped_images = [(image_path, base_name)]
                    result = ocr_model.ocr(image_path, cls=True)
                    inner_result = result[0] if result else []
                    text_blocks = []
                    for res in inner_result:
                        if isinstance(res[0], list) and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in res[0]):
                            coords = [int(pt[0]) for pt in res[0]]
                            y_coords = [int(pt[1]) for pt in res[0]]
                            text_blocks.append((res[1][0], coords + y_coords))
                        else:
                            print(f"Skipping invalid box format in OCR result for {image_path}: {res[0]}")
                    has_surrounding_text = False
                    if selected_crops:
                        for crop, conf, box_idx in selected_crops:
                            x_min = int(max(0, x_center - bbox_width / 2))
                            y_min = int(max(0, y_center - bbox_height / 2))
                            x_max = int(min(width, x_center + bbox_width / 2))
                            y_max = int(min(height, y_center + bbox_height / 2))
                            for text, coords in text_blocks:
                                if text.strip() and len(text.split()) > 1:
                                    if len(coords) >= 4:
                                        x_text_min = min(coords[:4])
                                        x_text_max = max(coords[:4])
                                        y_text_min = min(coords[4:])
                                        y_text_max = max(coords[4:])
                                        if (x_text_min < x_min or x_text_max > x_max or y_text_min < y_min or y_text_max > y_max):
                                            has_surrounding_text = True
                                            break
                                if has_surrounding_text:
                                    break
                    if has_surrounding_text:
                        status_placeholder.info("Detected text around ID card, processing entire image without cropping.")
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        cropped_images = [(image_path, base_name)]
                    else:
                        cropped_images = []
                        crop_folder = os.path.join(output_dir, "cropped")
                        os.makedirs(crop_folder, exist_ok=True)
                        for crop, conf, box_idx in selected_crops:
                            if len(selected_crops) == 1:
                                crop_filename = f"{os.path.splitext(filename)[0]}.jpg"
                            else:
                                crop_filename = f"{os.path.splitext(filename)[0]}_conf{conf:.2f}_box{box_idx}.jpg"
                            crop_path = os.path.join(crop_folder, crop_filename)
                            cv2.imwrite(crop_path, crop)
                            cropped_images.append((crop_path, os.path.splitext(crop_filename)[0]))
                        if not cropped_images:
                            status_placeholder.info(f"No crops saved from {filename} with confidence >= 0.7.")
                            cropped_images = [(image_path, base_name)]


        status_placeholder.info("Image alignment has started...")
        aligned_dir = os.path.join(output_dir, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        api = aspose.ocr.AsposeOcr()
        aligned_images = []
        for crop_path, base_name in cropped_images:
            if not os.path.exists(crop_path):
                st.warning(f"Image not found at {crop_path}, skipping alignment.")
                continue
            ocr_input = aspose.ocr.OcrInput(aspose.ocr.InputType.SINGLE_IMAGE)
            ocr_input.add(crop_path)
            angles = api.calculate_skew(ocr_input)
            img_cv = cv2.imread(crop_path)
            if img_cv is None:
                st.warning(f"Failed to load image at {crop_path}, skipping alignment.")
                continue
            (h, w) = img_cv.shape[:2]
            angle = angles[0].angle
            angle_rad = math.radians(abs(angle))
            new_w = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))
            new_h = int(abs(w * math.sin(angle_rad)) + abs(h * math.cos(angle_rad)))
            center = (w // 2, h // 2)
            new_center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += new_center[0] - center[0]
            M[1, 2] += new_center[1] - center[1]
            rotated = cv2.warpAffine(img_cv, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            output_path = os.path.join(aligned_dir, os.path.basename(crop_path))
            cv2.imwrite(output_path, rotated)
            aligned_images.append((output_path, base_name))
        status_placeholder.info("Image alignment has completed.")


        status_placeholder.info("Text extraction has started...")
        ocr_dir = os.path.join(Config.TEMP_DIR, "paddle_result")
        image_ocr_dir = os.path.join(ocr_dir, "images")
        json_ocr_dir = os.path.join(ocr_dir, "annotations")
        os.makedirs(image_ocr_dir, exist_ok=True)
        os.makedirs(json_ocr_dir, exist_ok=True)
        for file in glob.glob(os.path.join(json_ocr_dir, "*.json")):
            os.remove(file)
        for file in glob.glob(os.path.join(image_ocr_dir, "*.jpg")):
            os.remove(file)
        json_data = []
        for img_file, base_name in aligned_images:
            result = ocr_model.ocr(img_file, cls=True)
            inner_result = result[0] if result else []
            boxes = [res[0] for res in inner_result]
            texts = [res[1][0] for res in inner_result]
            for box, text in zip(boxes, texts):
                coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                json_data.append({"text": text, "coordinates": coordinates})
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = draw_ocr(img, boxes, texts, None, font_path=Config.FONT_PATH)
            annotated_img = Image.fromarray(annotated)
            annotated_image_path = os.path.join(image_ocr_dir, f"{base_name}.jpg")
            annotated_img.save(annotated_image_path)
            json_output_path = os.path.join(json_ocr_dir, f"{base_name}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            json_data = []
        status_placeholder.info("Text extraction has completed.")


        status_placeholder.info("PII detection has started...")
        pii_dir = os.path.join(output_dir, "pii")
        os.makedirs(pii_dir, exist_ok=True)
        pii_result = {}
        for img_file, base_name in aligned_images:
            json_file = os.path.join(json_ocr_dir, f"{base_name}.json")
            image_path = os.path.join(image_ocr_dir, f"{base_name}.jpg")
            if os.path.exists(json_file) and os.path.exists(image_path):
                pii_result[base_name] = PIIExtractor.process_json_file(json_file, image_path)
                pii_json_path = os.path.join(pii_dir, f"{base_name}_pii.json")
                with open(pii_json_path, 'w', encoding='utf-8') as f:
                    json.dump(pii_result[base_name], f, ensure_ascii=False, indent=4)
            else:
                st.warning(f"JSON or image file missing for {base_name}, skipping PII detection.")
        status_placeholder.info("PII detection has completed.")


        status_placeholder.info("PII masking has started...")
        masked_dir = os.path.join(output_dir, "masked")
        os.makedirs(masked_dir, exist_ok=True)
        masked_images = []
        SHRINK_OFFSET = 3
        for img_file, base_name in aligned_images:
            json_file = os.path.join(json_ocr_dir, f"{base_name}.json")
            aligned_image_path = img_file
            if not os.path.exists(aligned_image_path):
                st.error(f"Aligned image not found at {aligned_image_path}. Skipping masking for {base_name}.")
                continue
            if not os.path.exists(json_file):
                st.error(f"JSON file not found at {json_file}. Skipping masking for {base_name}.")
                continue
            data = pii_result.get(base_name, {})
            try:
                image = Image.open(aligned_image_path).convert("RGB")
                masked_image = image.copy()
                draw = ImageDraw.Draw(masked_image)
            except Exception as e:
                st.error(f"Error opening aligned image {aligned_image_path}: {str(e)}")
                continue
            unique_coords = set()
            coordinates_found = False
            for field, info in data.items():
                if isinstance(info, dict) and info.get("value") != "NONE" and info.get("coordinates"):
                    for coord_set in info["coordinates"]:
                        if coord_set:
                            coord_tuple = tuple(tuple(point.items()) for point in coord_set)
                            unique_coords.add(coord_tuple)
            for coord_tuple in unique_coords:
                coord_set = [dict(point) for point in coord_tuple]
                x_coords = [point["x"] for point in coord_set]
                y_coords = [point["y"] for point in coord_set]
                left = max(0, min(x_coords))
                top = max(0, min(y_coords))
                right = min(image.width, max(x_coords))
                bottom = min(image.height, max(y_coords))
                if right > left and bottom > top:
                    left_shrunk = left + SHRINK_OFFSET
                    top_shrunk = top + SHRINK_OFFSET
                    right_shrunk = right - SHRINK_OFFSET
                    bottom_shrunk = bottom - SHRINK_OFFSET
                    if right_shrunk > left_shrunk and bottom_shrunk > top_shrunk:
                        for offset in range(1):
                            draw.rectangle(
                                [left_shrunk - offset, top_shrunk - offset, right_shrunk + offset, bottom_shrunk + offset],
                                outline="black",
                                width=0
                            )
                        draw.rectangle([left_shrunk, top_shrunk, right_shrunk, bottom_shrunk], fill="black")
                        coordinates_found = True
            output_path = os.path.join(masked_dir, f"{base_name}.jpg")
            try:
                if coordinates_found:
                    masked_image.save(output_path)
                    st.info(f"Saved masked image to {output_path}")
                else:
                    shutil.copy(aligned_image_path, output_path)
                    st.warning(f"No coordinates found for {base_name}, copied aligned image to {output_path}")
                masked_images.append((output_path, base_name))
            except Exception as e:
                st.error(f"Error saving or copying masked image to {output_path}: {str(e)}")
                continue
        status_placeholder.info("PII masking has completed.")
        return pii_result, masked_images

class StreamlitApp:
    """Class to handle the Streamlit interface and application logic."""
    def __init__(self):
        Config.setup_temp_dir()
        Config.validate_config()  # Validate config at initialization
        self.yolo_model, self.ocr_model = ModelLoader.load_models()
        self.initialize_session_state()
        self.setup_ui()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'pii_data' not in st.session_state:
            st.session_state.pii_data = {}
        if 'image_names' not in st.session_state:
            st.session_state.image_names = []
        if 'original_file_info' not in st.session_state:
            st.session_state.original_file_info = {}
        if 'show_pii' not in st.session_state:
            st.session_state.show_pii = False

    def reset_upload(self):
        """Reset session state to allow uploading new images."""
        st.session_state.processed_images = []
        st.session_state.pii_data = {}
        st.session_state.image_names = []
        st.session_state.original_file_info = {}
        st.session_state.show_pii = False

    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(Config.TEMP_DIR):
            shutil.rmtree(Config.TEMP_DIR)
            os.makedirs(Config.TEMP_DIR, exist_ok=True)

    def setup_ui(self):
        """Set up the Streamlit UI."""
        st.set_page_config(page_title="PII Masking App", layout="wide")
        st.markdown(
            """
            <style>
            .title-container {
                text-align: center;
                padding: 30px;
                background: linear-gradient(135deg, #007bff, #0056b3);
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
            }
            .title {
                font-size: 40px;
                font-weight: bold;
                color: #ffffff;
                font-family: 'Arial', sans-serif;
            }
            .description {
                font-size: 16px;
                color: #333333;
                margin-top: 10px;
                font-family: 'Arial', sans-serif;
            }
            .stButton>button {
                background-color: #007bff;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
            .stProgress .st-bo {
                background-color: #28a745;
            }
            .image-caption {
                font-size: 14px;
                color: #555555;
                margin-top: 5px;
                font-family: 'Arial', sans-serif;
            }
            .pii-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="title-container"><span class="title">PII Masking Application</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="description">Upload images or PDFs (JPG, JPEG, PNG, BMP, TIFF, PDF) to detect and mask Personally Identifiable Information (PII).</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload images or PDFs", type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf", "tif"], accept_multiple_files=True)
        if st.button("Process Images"):
            self.process_uploaded_files(uploaded_files)
        self.display_results()

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and perform PII masking."""
        self.reset_upload()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        gemini_count = 0
        batch_start_time = time.time()
        failed_files = []
        for i, file in enumerate(uploaded_files):
            if gemini_count >= Config.API_CALLS_PER_MINUTE:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time < Config.SECONDS_PER_MINUTE:
                    status_placeholder.info(f"Rate limit reached. Waiting for {Config.SECONDS_PER_MINUTE - elapsed_time:.2f} seconds...")
                    time.sleep(Config.SECONDS_PER_MINUTE - elapsed_time)
                batch_start_time = time.time()
                gemini_count = 0
            if isinstance(file, str):
                image_path = file
                original_file_name = os.path.basename(file)
                page_number = None
                image_paths = [(image_path, None)]
            else:
                temp_path = os.path.join(Config.TEMP_DIR, file.name)
                os.makedirs(Config.TEMP_DIR, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                image_path = temp_path
                original_file_name = file.name
                if file.name.lower().endswith((".bmp", ".tiff", ".tif", ".pdf")):
                    if file.name.lower().endswith(".pdf"):
                        image_paths = ImageProcessor.convert_pdf_to_images(temp_path, Config.TEMP_DIR)
                    else:
                        img = Image.open(temp_path)
                        jpg_path = os.path.join(Config.TEMP_DIR, os.path.splitext(file.name)[0] + ".jpg")
                        ImageProcessor.convert_to_jpg(img, jpg_path)
                        image_paths = [(jpg_path, None)]
                else:
                    image_paths = [(image_path, None)]
            for img_path, page_number in image_paths:
                status_placeholder.info(f"Processing {os.path.basename(img_path)}")
                try:
                    pii_result, masked_images = ImageProcessor.process_image(
                        img_path, Config.TEMP_DIR, self.yolo_model, self.ocr_model, status_placeholder,
                        original_file_name=original_file_name, page_number=page_number
                    )
                    if not masked_images:
                        st.warning(f"No ID detected in {os.path.basename(img_path)}. Using original image.")
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        masked_images = [(img_path, base_name)]
                    for masked_path, base_name in masked_images:
                        st.session_state.processed_images.append(masked_path)
                        st.session_state.pii_data[base_name] = pii_result.get(base_name, {})
                        st.session_state.image_names.append(base_name)
                        st.session_state.original_file_info[base_name] = (original_file_name, page_number)
                    gemini_count += 1
                except Exception as e:
                    st.error(f"Error processing {os.path.basename(img_path)}: {str(e)}")
                    failed_files.append((os.path.basename(img_path), str(e)))
                    continue
                progress_bar.progress((i + 1) / len(uploaded_files))
            if failed_files:
                os.makedirs(Config.TEMP_DIR, exist_ok=True)
                log_file_path = os.path.join(Config.TEMP_DIR, "failed_files.log")
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    log_file.write("Files that failed processing:\n")
                    for file_name, reason in failed_files:
                        log_file.write(f"{file_name}: {reason}\n")
                st.warning(f"Some files failed to process. Check '{log_file_path}' for details.")

    def display_results(self):
        """Display processed images and PII data."""
        if st.session_state.processed_images:
            st.header("Masked Images")
            output_format = st.selectbox("Select download format:", ["JPG", "PNG", "PDF"])
            if st.button("Show PII Fields"):
                st.session_state.show_pii = not getattr(st.session_state, "show_pii", False)
            pii_summary = {}
            for i, (masked_path, base_name) in enumerate(zip(st.session_state.processed_images, st.session_state.image_names)):
                st.subheader(f"Image: {base_name}")
                if not os.path.exists(masked_path):
                    st.error(f"Masked image not found at {masked_path}. Check the masking step.")
                    continue
                img = Image.open(masked_path)
                st.image(img, caption=f"Masked {base_name}", use_container_width=True)
                if getattr(st.session_state, "show_pii", False):
                    pii_fields = st.session_state.pii_data.get(base_name, {})
                    st.write("**PII Fields**:")
                    pii_summary[base_name] = {}
                    for field, data in pii_fields.items():
                        if isinstance(data, dict) and data.get("value") != "NONE":
                            st.write(f"{field}: {data['value']}")
                            pii_summary[base_name][field] = data['value']
            pii_buffer = io.BytesIO()
            pii_json = json.dumps(pii_summary, ensure_ascii=False, indent=4)
            pii_buffer.write(pii_json.encode('utf-8'))
            st.download_button(
                label="Download PII Data as JSON",
                data=pii_buffer.getvalue(),
                file_name="pii_summary.json",
                mime="application/json"
            )
            format_map = {"JPG": "JPEG", "PNG": "PNG", "PDF": "PDF"}
            if len(st.session_state.processed_images) == 1:
                img = Image.open(st.session_state.processed_images[0])
                buf = io.BytesIO()
                img.save(buf, format=format_map[output_format])
                ext = output_format.lower() if output_format != "JPG" else "jpg"
                st.download_button(
                    label=f"Download {st.session_state.image_names[0]}.{ext}",
                    data=buf.getvalue(),
                    file_name=f"{st.session_state.image_names[0]}.{ext}",
                    mime=f"image/{ext}" if ext != "pdf" else "application/pdf"
                )
            else:
                zip_buffer = io.BytesIO()
                if output_format == "PDF":
                    pdf_images = [Image.open(path).convert("RGB") for path in st.session_state.processed_images]
                    if pdf_images:
                        pdf_images[0].save(
                            zip_buffer,
                            format="PDF",
                            save_all=True,
                            append_images=pdf_images[1:]
                        )
                        st.download_button(
                            label="Download all as PDF",
                            data=zip_buffer.getvalue(),
                            file_name="masked_images.pdf",
                            mime="application/pdf"
                        )
                else:
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for path, base_name in zip(st.session_state.processed_images, st.session_state.image_names):
                            img = Image.open(path)
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format=format_map[output_format])
                            zf.writestr(f"{base_name}.{output_format.lower()}", img_buffer.getvalue())
                        zf.writestr("pii_summary.json", pii_json.encode('utf-8'))
                    st.download_button(
                        label=f"Download all as ZIP ({output_format})",
                        data=zip_buffer.getvalue(),
                        file_name=f"masked_images_{output_format.lower()}.zip",
                        mime="application/zip"
                    )
        st.button("Back to Upload", on_click=self.reset_upload)
        st.button("Clear Temporary Files", on_click=self.cleanup)

if __name__ == "__main__":
    app = StreamlitApp()
