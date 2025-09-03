import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import zipfile
import tempfile
import traceback
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
from aspose.ocr import AsposeOcr, OcrInput, InputType
import math
from paddleocr import PaddleOCR, draw_ocr
import pdf2image
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
import os


class Config:
    TEMP_DIR = os.path.join(os.path.dirname(__file__), "pii_masking_temp")
    GOOGLE_DRIVE_FILE_ID = "1BUHPtTeZp1vDTfuXX86qPtceYxNxdbp4"
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "GEMINI_API_KEY")
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    API_CALLS_PER_MINUTE = 4
    SECONDS_PER_MINUTE = 60
   
    keywords = [
        "Name", "Husband", "Husband's Name", "Mother's Name", "DOB", "D.O.B", "Date of Birth",
        "Date of Birth/Age", "Address", "YoB", "Year of Birth", "S/W/D", "Licence No",
        "Number", "Son/Daughter/Wife of", "DL NO", "Father", "Son of", "No.", "DL",
        "Driving Licence No", "S/W/D of", "Father's Name", "Permanent Account Number",
        "Passport No.", "Surname", "Given Name(s)", "Place of Birth", "Elector's Name",
        "Relation's Name", "Mobile No.", "Phone Number", "Phone No.", "Mobile Number",
        "Bill No.", "Zipcode"
    ]


    @staticmethod
    def setup_temp_dir():
        """Create the temporary directory if it doesn't exist."""
        os.makedirs(Config.TEMP_DIR, exist_ok=True)


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
    def sanitize_filename(filename: str) -> str:
        return re.sub(r'[^\w\-_\.]', '_', filename)
    

    @staticmethod
    def encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            st.error(f"Failed to encode image {image_path}: {str(e)}")
            return None

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
        for keyword in Config.keywords:
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
            text_safe = text.replace('{', '{{').replace('}', '}}')
            json_data_str = json.dumps(json_data, ensure_ascii=False).replace('{', '{{').replace('}', '}}')
            prompt = (
                    "You are tasked with extracting Personally Identifiable Information (PII) from documents, including mixed content such as text reports and embedded ID cards (e.g., Aadhaar Card, PAN Card, Passport, Driving License, Voter ID Card). Use the provided text and JSON annotations, with an optional image to improve text quality and PII detection. If an image is provided, use it only to enhance text quality and assist in identifying PII fields, not for extracting text coordinates. Coordinates for PII fields must be sourced exclusively from the provided JSON annotations. Do not extract or adjust coordinates based on the image.\n\n"
                    "The document may contain a mix of free text (e.g., reports, acknowledgments) and embedded ID cards. Identify the document type as 'Mixed' if both text and an ID card are present, or classify the specific ID card type (e.g., 'PAN Card') if an ID card is dominant, or 'Text Only' if no ID card is detected. Extract PII from both the free text and any embedded ID card.\n\n"
                    "**Extracting All Names**: Identify all distinct names in the document that are not labeled as 'Father's Name', 'Mother's Name', or 'Husband's Name'. The primary name (e.g., the report author or main individual) should be assigned to the 'Name' field. For additional names (e.g., mentors, teammates), create fields named 'Name1', 'Name2', etc., in the order they appear. A name is valid if it contains only alphabetic characters, spaces, or '&', and is at least 3 characters long. Include coordinates from all JSON annotation text blocks that contain the name or any part of it, using case-insensitive and space-insensitive matching.\n\n"
                    "If an image is provided, use it only to improve text quality to correct OCR errors and identify PII fields, not for extracting text coordinates. "
                    "Coordinates for PII fields must be sourced exclusively from the provided JSON annotations. Do not extract or adjust coordinates based on the image. The output must preserve the format of the coordinates as a list of dictionaries, each containing 'x' and 'y' keys, exactly as provided in the annotation. For example, if the annotation is: "
                    "{\n"
                    " \"text\": \"VUAY KUMAR\",\n"
                    " \"coordinates\": [\n"
                    " {\"x\": 29, \"y\": 283},\n"
                    " {\"x\": 123, \"y\": 285},\n"
                    " {\"x\": 122, \"y\": 303},\n"
                    " {\"x\": 29, \"y\": 301}\n"
                    " ]\n"
                    "}\n"
                    "Then the output must be:\n"
                    "\"Name\": {\n"
                    " \"value\": \"VIJAY KUMAR\",\n"
                    " \"coordinates\": [\n"
                    " {\"x\": 29, \"y\": 283},\n"
                    " {\"x\": 123, \"y\": 285},\n"
                    " {\"x\": 122, \"y\": 303},\n"
                    " {\"x\": 29, \"y\": 301}\n"
                    " ]\n"
                    "}\n"
                    "Ensure coordinates are copied verbatim from the annotation, with no alterations to values or structure. For multiple coordinates like:\n"
                    "{\n"
                    " \"text\": \"SUMRA\",\n"
                    " \"coordinates\": [\n"
                    " {\"x\": 292, \"y\": 140},\n"
                    " {\"x\": 360, \"y\": 146},\n"
                    " {\"x\": 358, \"y\": 166},\n"
                    " {\"x\": 290, \"y\": 160}\n"
                    " ]\n"
                    "}\n"
                    "and\n"
                    "{\n"
                    " \"text\": \"SAJID UMAR\",\n"
                    " \"coordinates\": [\n"
                    " {\"x\": 290, \"y\": 191},\n"
                    " {\"x\": 421, \"y\": 199},\n"
                    " {\"x\": 420, \"y\": 220},\n"
                    " {\"x\": 289, \"y\": 211}\n"
                    " ]\n"
                    "}\n"
                    "Then the output should be:\n"
                    "\"Name\": {\n"
                    " \"value\": \"SUMRA SAJID UMAR\",\n"
                    " \"coordinates\": [\n"
                    " {\"x\": 292, \"y\": 140},\n"
                    " {\"x\": 360, \"y\": 146},\n"
                    " {\"x\": 358, \"y\": 166},\n"
                    " {\"x\": 290, \"y\": 160},\n"
                    " {\"x\": 290, \"y\": 191},\n"
                    " {\"x\": 421, \"y\": 199},\n"
                    " {\"x\": 420, \"y\": 220},\n"
                    " {\"x\": 289, \"y\": 211}\n"
                    " ]\n"
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
                    " \"Document Type\": \"Mixed|Text Only|Aadhaar Card|PAN Card|Passport|Driving License|Voter ID Card\",\n"
                    " \"Name\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name1\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name2\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name3\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name4\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name5\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name6\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name7\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name8\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name9\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Name10\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Father's Name\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Mother's Name\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Husband's Name\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Phone Number\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Date of Birth\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Aadhaar ID\": {\"value\": null, \"coordinates\": []},\n"
                    " \"PAN ID\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Passport ID\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Driving License ID\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Voter ID\": {\"value\": null, \"coordinates\": []},\n"
                    " \"Address\": {\"value\": null, \"coordinates\": []},\n"
                    " \"ZIP Code\": {\"value\": null, \"coordinates\": []}\n"
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
                    response = requests.post(Config.GEMINI_API_URL, json=payload, headers=headers, timeout=30)
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
    def process_image(image_path: str, output_dir: str, yolo_model, ocr_model, status_placeholder, original_file_name: str, page_number: Optional[int] = None) -> tuple[Dict, List[tuple]]:
        """Process an image to detect and mask PII."""
        # Step 1: Use input image directly
        status_placeholder.info("ID card detection has started...")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_base_name = HelperFunctions.sanitize_filename(base_name)
       
        # Step 2: YOLOv8 Detection
        status_placeholder.info(f"Processing {os.path.basename(image_path)}: The ID detection has started")
        results = yolo_model.predict(
            source=image_path,
            conf=st.session_state.yolo_conf,
            iou=0.7,
            save=True,
            save_txt=True,
            save_conf=True,
            project=os.path.join(output_dir, "yolo_output"),
            name="predict"
        )
        print(f"YOLO results for {image_path}: {results}")
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
                # Read image and prepare for cropping
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                with open(label_path, 'r') as file:
                    lines = [line.strip() for line in file.readlines() if len(line.strip().split()) == 6]
                if not lines:
                    status_placeholder.info("No valid bounding boxes detected, using original image.")
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    cropped_images = [(image_path, base_name)]
                else:
                    # Process bounding boxes using your cropping logic
                    crops_info = []
                    for idx, line in enumerate(lines):
                        class_id, x_c, y_c, w, h, conf = map(float, line.split())
                        if conf < 0.7: # Apply confidence threshold
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
                    # Decide on cropping based on number of boxes
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
                        # Proceed to text detection to confirm
                    # Check for surrounding text using OCR on the original image
                    result = ocr_model.ocr(image_path, cls=True)
                    inner_result = result[0] if result else []
                    text_blocks = []
                    for res in inner_result:
                        if isinstance(res[0], list) and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in res[0]):
                            coords = [int(pt[0]) for pt in res[0]] # Extract x coordinates
                            y_coords = [int(pt[1]) for pt in res[0]] # Extract y coordinates
                            text_blocks.append((res[1][0], coords + y_coords)) # Combine x and y coords into a flat list
                        else:
                            print(f"Skipping invalid box format in OCR result for {image_path}: {res[0]}")
                    has_surrounding_text = False
                    if selected_crops:
                        for crop, conf, box_idx in selected_crops:
                            # Get bounding box coordinates
                            x_min = int(max(0, x_center - bbox_width / 2))
                            y_min = int(max(0, y_center - bbox_height / 2))
                            x_max = int(min(width, x_center + bbox_width / 2))
                            y_max = int(min(height, y_center + bbox_height / 2))
                           
                            for text, coords in text_blocks:
                                if text.strip() and len(text.split()) > 1: # Significant text
                                    if len(coords) >= 4: # Minimum for a quadrilateral
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
                        # Save cropped images
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
       
        # Step 3: Align Images
        status_placeholder.info("Image alignment has started...")
        aligned_dir = os.path.join(output_dir, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        api = AsposeOcr()
        aligned_images = []
        for crop_path, base_name in cropped_images:
            if not os.path.exists(crop_path):
                st.warning(f"Image not found at {crop_path}, skipping alignment.")
                continue
            ocr_input = OcrInput(InputType.SINGLE_IMAGE)
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
    
        # Step 3: OCR with PaddleOCR on aligned images
        status_placeholder.info("Text extraction has started...")
        ocr_dir = os.path.join(Config.TEMP_DIR, "paddle_result", original_file_name)
        image_ocr_dir = os.path.join(ocr_dir, "images")
        json_ocr_dir = os.path.join(ocr_dir, "annotations")
        os.makedirs(image_ocr_dir, exist_ok=True)
        os.makedirs(json_ocr_dir, exist_ok=True)

        # Clear previous OCR results for this image
        for file in glob.glob(os.path.join(json_ocr_dir, "*.json")):
            os.remove(file)
        for file in glob.glob(os.path.join(image_ocr_dir, "*.jpg")):
            os.remove(file)

        json_data = []
        boxes_all = []
        texts_all = []

        for img_file, base_name in aligned_images:
            try:
                print(f"Running OCR on {img_file}")
                result = ocr_model.ocr(img_file, cls=True)
                print(f"OCR result for {base_name}: {result}")
                if not result or not result[0]:
                    status_placeholder.error(f"OCR failed for {base_name}. Falling back to original image.")
                    result = ocr_model.ocr(image_path, cls=True)
                    print(f"OCR result for original image {base_name}: {result}")
                    if not result or not result[0]:
                        status_placeholder.error(f"OCR failed for original image {base_name}. Skipping text extraction.")
                        continue
                
                inner_result = result[0]
                for res in inner_result:
                    box, (text, _) = res
                    # Try to split based on keywords
                    keyword, split_idx = HelperFunctions.find_keyword_split(text)
                    if keyword and split_idx < len(text):
                        # Split text after the keyword
                        part1 = text[:split_idx].strip()
                        part2 = text[split_idx:].strip()
                        if len(part1) > 0 and len(part2) > 0:
                            total_len = len(part1) + len(part2)
                            ratio = len(part1) / total_len
                            split_boxes = HelperFunctions.split_box(box, ratio)
                            for part_text, part_box in zip([part1, part2], split_boxes):
                                coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in part_box]
                                json_data.append({"text": part_text, "coordinates": coordinates})
                                boxes_all.append(part_box)
                                texts_all.append(part_text)
                        else:
                            # Fallback if splitting results in empty parts
                            coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                            json_data.append({"text": text, "coordinates": coordinates})
                            boxes_all.append(box)
                            texts_all.append(text)
                    elif ':' in text:
                        # If no keyword, try splitting on ':'
                        parts = text.split(':', 1)
                        parts = [part.strip() for part in parts]
                        if len(parts[0]) > 0 and len(parts[1]) > 0:
                            total_len = len(parts[0]) + len(parts[1])
                            ratio = len(parts[0]) / total_len
                            split_boxes = HelperFunctions.split_box(box, ratio)
                            for part_text, part_box in zip(parts, split_boxes):
                                coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in part_box]
                                json_data.append({"text": part_text, "coordinates": coordinates})
                                boxes_all.append(part_box)
                                texts_all.append(part_text)
                        else:
                            # Fallback if splitting on ':' fails
                            coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                            json_data.append({"text": text, "coordinates": coordinates})
                            boxes_all.append(box)
                            texts_all.append(text)
                    else:
                        # No keyword or ':', use the full text and box
                        coordinates = [{"x": int(pt[0]), "y": int(pt[1])} for pt in box]
                        json_data.append({"text": text, "coordinates": coordinates})
                        boxes_all.append(box)
                        texts_all.append(text)

                # Draw annotated image
                img = cv2.imread(img_file)
                if img is None:
                    status_placeholder.error(f"Failed to load image {img_file} for annotation.")
                    print(f"Image loading failed for {img_file}: file exists={os.path.exists(img_file)}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                font_path = "C:/Windows/Fonts/arial.ttf"
                if not os.path.exists(font_path):
                    print(f"Font file {font_path} not found, using default font")
                    font_path = None
                annotated = draw_ocr(img, boxes_all, texts_all, None, font_path=font_path)
                annotated_img = Image.fromarray(annotated)
                annotated_path = os.path.join(image_ocr_dir, f"{base_name}_annotated.jpg")
                annotated_img.save(annotated_path)
                print(f"Saved annotated image to {annotated_path}, exists={os.path.exists(annotated_path)}")

                # Save JSON
                json_output_path = os.path.join(json_ocr_dir, f"{base_name}.json")
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                print(f"Saved JSON to {json_output_path}, exists={os.path.exists(json_output_path)}")
                json_data = []  # Clear for next image
                boxes_all = []
                texts_all = []

            except Exception as e:
                status_placeholder.error(f"Error processing {base_name} in OCR: {str(e)}")
                print(f"OCR exception for {base_name}: {str(e)}")
                continue

        status_placeholder.info("Text extraction has completed.")
       
        # Step 5: PII Detection
        status_placeholder.info("PII detection has started...")
        pii_dir = os.path.join(output_dir, "pii")
        os.makedirs(pii_dir, exist_ok=True)
        pii_result = {}
        for img_file, base_name in aligned_images:
            json_file = os.path.join(json_ocr_dir, f"{base_name}.json")
            image_path = os.path.join(image_ocr_dir, f"{base_name}_annotated.jpg")
            print(f"Checking JSON file: {json_file}, exists={os.path.exists(json_file)}")
            print(f"Checking image file: {image_path}, exists={os.path.exists(image_path)}")
            if os.path.exists(json_file) and os.path.exists(image_path):
                pii_result[base_name] = PIIExtractor.process_json_file(json_file, image_path)
                pii_json_path = os.path.join(pii_dir, f"{base_name}_pii.json")
                with open(pii_json_path, 'w', encoding='utf-8') as f:
                    json.dump(pii_result[base_name], f, ensure_ascii=False, indent=4)
                print(f"Saved PII JSON to {pii_json_path}, exists={os.path.exists(pii_json_path)}")
            else:
                st.warning(f"JSON or image file missing for {base_name}, skipping PII detection.")
        status_placeholder.info("PII detection has completed.")
       
        # Step 6: Mask PII
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
            output_path = os.path.join(masked_dir, f"{base_name}_masked.jpg")
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
    def __init__(self):
        Config.setup_temp_dir()
        self.yolo_model, self.ocr_model = ModelLoader.load_models()
        self.initialize_session_state()
        self.setup_ui()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'pii_data' not in st.session_state:
            st.session_state.pii_data = {}
        if 'image_names' not in st.session_state:
            st.session_state.image_names = []
        if 'show_pii' not in st.session_state:
            st.session_state.show_pii = False
        if 'original_file_info' not in st.session_state:
            st.session_state.original_file_info = {}
        # Initialize internal state for processing preview checkboxes, defaulting to False
        if 'show_yolo' not in st.session_state:
            st.session_state.show_yolo = False
        if 'show_cropped' not in st.session_state:
            st.session_state.show_cropped = False
        if 'show_aligned' not in st.session_state:
            st.session_state.show_aligned = False
        if 'show_text' not in st.session_state:
            st.session_state.show_text = False
        if 'show_pii_detect' not in st.session_state:
            st.session_state.show_pii_detect = False
        if 'show_masked' not in st.session_state:
            st.session_state.show_masked = False
        # Initialize uploaded_files in session state for file_uploader interaction
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []


    def cleanup(self):
        """Remove temporary files and reset directories."""
        if os.path.exists(Config.TEMP_DIR):
            shutil.rmtree(Config.TEMP_DIR)
            os.makedirs(Config.TEMP_DIR, exist_ok=True)
        st.success("Temporary files cleared successfully!")


    def reset_upload(self):
        """Reset session state to allow uploading new images."""
        st.session_state.processed_images = []
        st.session_state.pii_data = {}
        st.session_state.image_names = []
        st.session_state.original_file_info = {}
        st.session_state.show_pii = False
        st.session_state.show_yolo = False 
        st.session_state.show_cropped = False
        st.session_state.show_aligned = False
        st.session_state.show_text = False
        st.session_state.show_pii_detect = False
        st.session_state.show_masked = False
        st.session_state.uploaded_files = []

    def setup_ui(self):
        """Set up the Streamlit UI."""
        st.set_page_config(page_title="PII Masking App", layout="wide")
        st.markdown(
            """
            <style>
            /* Overall App Styling */
            .stApp {
                background-color: #f0f2f6; /* Light grey background for a clean look */
                font-family: 'Segoe UI', sans-serif;
            }
            /* Header/Title Section */
            .title-container {
                text-align: center;
                padding: 40px;
                background: linear-gradient(135deg, #2c3e50, #4a69bd); /* Deep blue-grey to moderate blue */
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
                margin-bottom: 40px;
                animation: fadeIn 1.5s ease-out;
            }
            .title {
                font-size: 56px;
                font-weight: 800;
                color: #ecf0f1; /* Off-white for contrast */
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.6);
                letter-spacing: 1.5px;
                animation: slideInLeft 1s ease-out;
            }
            .description {
                font-size: 20px;
                color: #bdc3c7; /* Lighter grey for description */
                margin-top: 20px;
                font-family: 'Segoe UI', sans-serif;
                text-align: center;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
                animation: fadeIn 2s ease-out;
            }
            /* Buttons General Styling */
            .stButton>button {
                background-color: #28a745; /* Vibrant Green for primary actions */
                color: white;
                border-radius: 10px;
                padding: 14px 30px;
                font-size: 19px;
                font-weight: bold;
                transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
                border: none;
                cursor: pointer;
                box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
                margin: 5px; /* Spacing between buttons */
            }
            .stButton>button:hover {
                background-color: #218838; /* Darker green on hover */
                transform: translateY(-3px);
                box_shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            }
            .stButton>button:active {
                transform: translateY(0);
                box_shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }
            /* Specific Button Styles (if needed for distinction) */
            .stButton button[data-testid="stFileUploadDropzone"] {
                background-color: #007bff; /* Blue for file upload */
            }
            .stButton button[data-testid="stFileUploadDropzone"]:hover {
                background-color: #0056b3;
            }
            /* Style for the Clear Temporary Files button */
            .stButton button[key="clear_temp_button"] {
                background-color: #dc3545; /* Red for destructive action */
            }
            .stButton button[key="clear_temp_button"]:hover {
                background-color: #c82333;
            }
            /* Progress Bar */
            .stProgress .st-bo {
                background-color: #17a2b8; /* Teal-blue for progress */
                height: 10px;
                border-radius: 5px;
            }
            .stProgress .st-dm { /* Track */
                background-color: #e0e0e0;
                border-radius: 5px;
            }
            /* Image Display */
            .stImage {
                border: 2px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .image-caption {
                font-size: 15px;
                color: #555555;
                margin-top: 10px;
                font-family: 'Segoe UI', sans-serif;
                text-align: center;
                font-style: italic;
            }
            /* PII Section */
            .pii-section {
                background-color: #e9f5fd; /* Light blue background for PII */
                padding: 25px;
                border-radius: 12px;
                margin-top: 10px; /* Adjusted margin */
                margin-bottom: 20px; /* Added margin */
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border-left: 6px solid #007bff; /* Stronger blue border */
                font-size: 16px;
                color: #333333;
                line-height: 1.8;
            }
            .pii-section h3 {
                color: #0056b3;
                margin-top: 0;
            }
            /* Sidebar Styling */
            .sidebar .sidebar-content {
                background-color: #e9ecef; /* Slightly darker light grey for sidebar */
                padding: 30px 20px;
                border-right: 2px solid #dcdcdc;
            }
            .sidebar .st-bb { /* Sidebar header */
                color: #2c3e50;
                font-weight: bold;
                font-size: 24px;
                margin-bottom: 20px;
            }
            /* Temporary Files Section in Sidebar */
            .temp-section {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                margin-top: 15px;
                margin-bottom: 15px; /* Added margin for separation */
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                border-left: 5px solid #ffc107; /* Warm amber for temp files */
            }
            .temp-section .stExpander {
                margin-top: 10px;
            }
            .temp-subheader {
                font-size: 19px;
                font-weight: bold;
                color: #333333;
                margin-bottom: 15px;
                font-family: 'Segoe UI', sans-serif;
                border-bottom: 1px dashed #cccccc;
                padding-bottom: 5px;
            }
            /* General Headings */
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50; /* Dark blue-grey for headings */
                font-family: 'Segoe UI', sans-serif;
                font-weight: 600;
            }
            .subheader {
                font-size: 32px;
                font-weight: bold;
                color: #007bff; /* Primary blue */
                margin-top: 40px;
                margin-bottom: 25px;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
                text-align: center;
            }
            /* File Uploader Customization */
            .stFileUploader {
                border: 2px dashed #007bff; /* Blue dashed border */
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                margin-bottom: 20px;
                background-color: #e6f3ff; /* Light blue background */
            }
            .stFileUploader label {
                font-size: 20px;
                font-weight: 600;
                color: #0056b3;
            }
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideInLeft {
                from { transform: translateX(-100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="title-container"><span class="title">🔒 AutoPII Mask</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="description">Upload your sensitive documents (images or PDFs) here. Our AI detects and masks Personally Identifiable Information (PII) to ensure your data privacy and compliance. Supporting JPG, JPEG, PNG, BMP, TIFF, and PDF formats.</div>', unsafe_allow_html=True)
        self.setup_sidebar()
        st.markdown("---") # Visual separator for main content
        
        st.write("") 
        st.write("") 
        col1, col2, col3 = st.columns([1, 2, 1]) # Use columns to center elements
        with col2:
            self.setup_file_uploader()
            st.markdown("<br>", unsafe_allow_html=True) # Add some space below uploader
            # Process button - more prominent and inviting
            if st.button("🚀 Process Files & Mask PII", key="process_files_button"):
                if not st.session_state.uploaded_files:
                    st.warning("🧐 Please upload at least one file to begin the processing.")
                else:
                    self.process_uploaded_files(st.session_state.uploaded_files)
        st.markdown("---") 
        self.display_results()


    def setup_sidebar(self):
        """Set up the sidebar for configuration and temporary files, with enhanced UI."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            st.session_state.yolo_conf = st.slider(
                "YOLO Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Adjust the confidence threshold for YOLO object detection."
            )
            st.markdown("---")
            st.header("🔬 Processing Preview")
            stages = [
                ("YOLO Output", "show_yolo", "yolo_output/predict*", "*.jpg", "📦 YOLO Output (ID Detection)"),
                ("Cropped Images", "show_cropped", "cropped", "*.jpg", "✂️ Cropped Images (Pre-Alignment)"),
                ("Aligned Images", "show_aligned", "aligned", "*.jpg", "📐 Aligned Images (Ready for OCR)"),
                ("Text Extraction", "show_text", "paddle_result/*/images", "*_annotated.jpg", "📝 Text Extraction Visualizations"),
                ("PII Detection", "show_pii_detect", None, None, "📊 Detected PII Data (Raw)"),
                ("Masked Images", "show_masked", "masked", "*.jpg", "🎭 Masked Images (Before Download)"),
            ]
            with st.expander("Show Detailed Processing Stages", expanded=False):
                # Initialize selected_preview_stages if not already set
                if 'selected_preview_stages' not in st.session_state:
                    st.session_state.selected_preview_stages = []
                # Callback function to update session state on multiselect change
                def update_preview_stages():
                    st.session_state.selected_preview_stages = st.session_state.preview_stages_multiselect
                    for stage_name, session_var, _, _, _ in stages:
                        st.session_state[session_var] = (stage_name in st.session_state.selected_preview_stages)
                    print(f"Updated selected_preview_stages: {st.session_state.selected_preview_stages}")
                # Multiselect with on_change callback
                st.multiselect(
                    "Choose stages to display for debugging:",
                    [s[0] for s in stages],
                    default=st.session_state.selected_preview_stages,
                    key="preview_stages_multiselect",
                    on_change=update_preview_stages,
                    help="Select specific intermediate processing stages to visualize. Selections should update immediately."
                )
            # Display content for selected stages
            if st.session_state.processed_images:
                for stage_name, session_var, subdir, pattern, nice_name in stages:
                    if st.session_state[session_var]:
                        st.markdown(f'<div class="temp-section">', unsafe_allow_html=True)
                        st.markdown(f'<span class="temp-subheader">{nice_name}</span>', unsafe_allow_html=True)
                        if session_var == "show_pii_detect":
                            if st.session_state.image_names:
                                for base_name in st.session_state.image_names:
                                    pii_data = st.session_state.pii_data.get(base_name, {})
                                    st.write(f"**PII for `{base_name}`**:")
                                    st.markdown('<div class="pii-section">', unsafe_allow_html=True)
                                    if pii_data:
                                        found_pii = False
                                        for field, value in pii_data.items():
                                            if value != "NONE":
                                                st.write(f"**{field}**: `{value}`")
                                                found_pii = True
                                        if not found_pii:
                                            st.info("No PII detected for this image.")
                                    else:
                                        st.info("No PII data available for this image.")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            if subdir:
                                if session_var == "show_yolo":
                                    yolo_base = os.path.join(Config.TEMP_DIR, "yolo_output")
                                    predict_dirs = sorted(glob.glob(os.path.join(yolo_base, "predict*")), key=os.path.getmtime, reverse=True)
                                    image_paths = []
                                    if predict_dirs:
                                        latest_predict_dir = predict_dirs[0]
                                        image_paths = glob.glob(os.path.join(latest_predict_dir, "*.jpg"))
                                        print(f"YOLO image paths: {image_paths}")
                                else:
                                    full_path = os.path.join(Config.TEMP_DIR, subdir)
                                    image_paths = glob.glob(os.path.join(full_path, pattern))
                                    print(f"{nice_name} image paths: {image_paths}")
                                if image_paths:
                                    for img_path in image_paths:
                                        if os.path.exists(img_path):
                                            try:
                                                img = Image.open(img_path)
                                                st.image(img, caption=f"{nice_name}: {os.path.basename(img_path)}", use_container_width=True)
                                            except Exception as e:
                                                st.error(f"Could not load image {os.path.basename(img_path)}: {e}")
                                        else:
                                            st.warning(f"Image file {img_path} does not exist.")
                                else:
                                    st.info(f"No {nice_name.lower()} images found yet for current session.")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Upload and process images to view detailed processing stages.")
            st.markdown("---")
            st.header("🗑️ Temporary Files")
            if st.button("Clear All Temporary Files 🧹", key="clear_temp_button"):
                self.cleanup()


    def setup_file_uploader(self):
        """Set up the file uploader."""
        # The key ensures the uploader resets when session state is cleared
        st.session_state.uploaded_files = st.file_uploader(
            "🖼️ Upload Your Images or PDFs",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf", "tif"],
            accept_multiple_files=True,
            key="file_uploader_widget", # Unique key for the widget
            help="Accepted image formats: JPG, JPEG, PNG, BMP, TIFF, TIF. Also supports PDF documents."
        )


    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and handle image processing."""
        self.reset_upload() # Ensure a clean slate before processing new files
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        # Rate-limiting variables - Logic unchanged
        gemini_count = 0
        batch_start_time = time.time()
        failed_files = []
        for i, file in enumerate(uploaded_files):
            # Rate-limiting check - Logic unchanged
            if gemini_count >= Config.API_CALLS_PER_MINUTE:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time < Config.SECONDS_PER_MINUTE:
                    status_placeholder.info(f"⏳ Rate limit reached. Waiting for {Config.SECONDS_PER_MINUTE - elapsed_time:.2f} seconds before continuing...")
                    time.sleep(Config.SECONDS_PER_MINUTE - elapsed_time)
                batch_start_time = time.time()
                gemini_count = 0
            temp_path = os.path.join(Config.TEMP_DIR, file.name)
            os.makedirs(Config.TEMP_DIR, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(file.read())
            image_path = temp_path
            original_file_name = file.name
            # Handle PDF and other image formats - Logic unchanged
            image_paths_to_process = []
            if file.name.lower().endswith((".bmp", ".tiff", ".tif", ".pdf")):
                if file.name.lower().endswith(".pdf"):
                    status_placeholder.info(f"🔄 Converting PDF: '{file.name}' to individual images...")
                    image_paths_to_process = ImageProcessor.convert_pdf_to_images(temp_path, Config.TEMP_DIR)
                else:
                    img = Image.open(temp_path)
                    jpg_path = os.path.join(Config.TEMP_DIR, os.path.splitext(file.name)[0] + ".jpg")
                    ImageProcessor.convert_to_jpg(img, jpg_path)
                    image_paths_to_process = [(jpg_path, None)]
            else:
                image_paths_to_process = [(image_path, None)]
            for img_path, page_number in image_paths_to_process:
                status_placeholder.info(f"✨ Processing: '{os.path.basename(img_path)}' for PII detection and masking...")
                try:
                    # Core processing call - Logic unchanged
                    pii_result, masked_images = ImageProcessor.process_image(
                        img_path, Config.TEMP_DIR, self.yolo_model, self.ocr_model, status_placeholder,
                        original_file_name=original_file_name, page_number=page_number
                    )
                   
                    # If no masked images are returned (e.g., no ID detected), use original image for display
                    if not masked_images:
                        st.warning(f"⚠️ No ID detected in '{os.path.basename(img_path)}'. Displaying original image without masking.")
                        base_name = HelperFunctions.sanitize_filename(os.path.splitext(os.path.basename(img_path))[0])
                        masked_images = [(img_path, base_name)] # Use the original image path for display
                    for masked_path, base_name in masked_images:
                        st.session_state.processed_images.append(masked_path)
                        # Ensure pii_result is correctly structured for the base_name
                        if base_name in pii_result:
                            st.session_state.pii_data[base_name] = pii_result[base_name]
                        else:
                            st.session_state.pii_data[base_name] = {} # No PII found for this specific image
                       
                        st.session_state.image_names.append(base_name)
                        st.session_state.original_file_info[base_name] = (original_file_name, page_number)
                   
                    gemini_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing '{os.path.basename(img_path)}': {str(e)}")
                    failed_files.append((os.path.basename(img_path), str(e)))
                    # Continue to next file even if one fails
                    continue
            progress_bar.progress((i + 1) / len(uploaded_files))
        status_placeholder.success("✅ All selected files processed! Scroll down to view results and download.")
        progress_bar.empty() # Clear the progress bar after completion
        # Log failed files - Logic unchanged
        if failed_files:
            os.makedirs(Config.TEMP_DIR, exist_ok=True)
            log_file_path = os.path.join(Config.TEMP_DIR, "failed_files.log")
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write("Files that failed processing:\n")
                for file_name, reason in failed_files:
                    log_file.write(f"{file_name}: {reason}\n")
            st.warning(f"Some files failed to process. Check '{log_file_path}' for details.")
    
    
    def display_results(self):
        """Display processed images and PII data with download options."""
        if st.session_state.processed_images:
            st.markdown("---")
            st.markdown('<div class="subheader">🖼️ Masked Images & PII Data</div>', unsafe_allow_html=True)
            st.write("Review the processed images and download the masked files along with a summary of detected PII.")
            col_display_options, col_download_format = st.columns([1, 1])
            with col_display_options:
                if st.button("👁️ Show/Hide Detected PII Details", key="toggle_pii_button", help="Toggle visibility of the extracted PII data for each image."):
                    st.session_state.show_pii = not st.session_state.show_pii
            with col_download_format:
                output_format = st.selectbox(
                    "💾 Select Download Image Format:",
                    ["JPG", "PNG", "PDF"],
                    index=0,
                    help="Choose the desired format for downloading your masked images."
                )
            st.markdown("<br>", unsafe_allow_html=True)
            pii_summary = {}
            for i, (masked_path, base_name) in enumerate(zip(st.session_state.processed_images, st.session_state.image_names)):
                st.markdown(f"### Image: **`{base_name}`**")
               
                if not os.path.exists(masked_path):
                    st.error(f"🚨 Masked image file not found at '{masked_path}'. This indicates an issue during the masking step for this particular image.")
                    continue
                   
                img = Image.open(masked_path)
                st.image(img, caption=f"Masked Image: {st.session_state.original_file_info.get(base_name, ('N/A', 'N/A'))[0]}{f' (Page {st.session_state.original_file_info[base_name][1]})' if st.session_state.original_file_info.get(base_name, ('N/A', 'N/A'))[1] is not None else ''}", use_container_width=True)
                if st.session_state.show_pii:
                    pii_fields = st.session_state.pii_data.get(base_name, {})
                    st.write("---")
                    st.markdown("#### Detected PII Fields:")
                    st.markdown('<div class="pii-section">', unsafe_allow_html=True)
                    if pii_fields:
                        pii_summary[base_name] = {}
                        found_pii = False
                        for field, value in pii_fields.items():
                            if value != "NONE":
                                st.markdown(f"**{field}**: `{value}`")
                                pii_summary[base_name][field] = value
                                found_pii = True
                            else:
                                st.markdown(f"**{field}**: `Not Detected`")
                        if not found_pii:
                            st.info("No PII detected for this image.")
                    else:
                        st.info("No PII data available for this image.")
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
            # Rest of the download logic remains unchanged
            st.markdown("### ⬇️ Download Options")
            col_json, col_download_files = st.columns([1, 1])
            with col_json:
                pii_buffer = io.BytesIO()
                pii_json = json.dumps(pii_summary, ensure_ascii=False, indent=4)
                pii_buffer.write(pii_json.encode('utf-8'))
                pii_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download All PII Data as JSON",
                    data=pii_buffer,
                    file_name="pii_summary.json",
                    mime="application/json",
                    help="Download a JSON file containing all detected PII across all processed images."
                )
               
            with col_download_files:
                format_map = {"JPG": "JPEG", "PNG": "PNG", "PDF": "PDF"}
                grouped_images = {}
                for masked_path, base_name in zip(st.session_state.processed_images, st.session_state.image_names):
                    original_file_name, page_number = st.session_state.original_file_info.get(base_name, (base_name, None))
                    if original_file_name not in grouped_images:
                        grouped_images[original_file_name] = []
                    grouped_images[original_file_name].append((Image.open(masked_path).convert("RGB"), page_number))
                   
                for original_name in grouped_images:
                    grouped_images[original_name].sort(key=lambda x: x[1] if x[1] is not None else 0)
                    grouped_images[original_name] = [img for img, _ in grouped_images[original_name]]
                if len(st.session_state.processed_images) == 1:
                    img = Image.open(st.session_state.processed_images[0]).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format=format_map[output_format])
                    ext = output_format.lower() if output_format != "JPG" else "jpg"
                    buf.seek(0)
                    st.download_button(
                        label=f"⬇️ Download Masked Image as {output_format}",
                        data=buf,
                        file_name=f"{st.session_state.image_names[0]}_masked.{ext}",
                        mime=f"image/{ext}" if ext != "pdf" else "application/pdf",
                        help=f"Download the single masked image in {output_format} format."
                    )
                else:
                    if output_format == "PDF":
                        for original_name, images_list in grouped_images.items():
                            if images_list:
                                pdf_combine_buffer = io.BytesIO()
                                if len(images_list) > 1:
                                    images_list[0].save(
                                        pdf_combine_buffer,
                                        format="PDF",
                                        save_all=True,
                                        append_images=images_list[1:]
                                    )
                                else:
                                    images_list[0].save(pdf_combine_buffer, format="PDF")
                                pdf_combine_buffer.seek(0)
                                st.download_button(
                                    label=f"⬇️ Download Masked '{os.path.splitext(original_name)[0]}' as PDF",
                                    data=pdf_combine_buffer,
                                    file_name=f"{os.path.splitext(original_name)[0]}_masked.pdf",
                                    mime="application/pdf",
                                    key=f"download_pdf_{original_name}"
                                )
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for path, base_name in zip(st.session_state.processed_images, st.session_state.image_names):
                                img = Image.open(path)
                                img_buffer = io.BytesIO()
                                img.save(img_buffer, format=format_map[output_format])
                                img_buffer.seek(0)
                                zf.writestr(f"{base_name}_masked.{output_format.lower()}", img_buffer.getvalue())
                                zf.writestr("pii_summary.json", pii_json.encode('utf-8'))
                        zip_buffer.seek(0)
                        st.download_button(
                            label=f"⬇️ Download All Masked Files as ZIP ({output_format})",
                            data=zip_buffer,
                            file_name=f"masked_data_{output_format.lower()}.zip",
                            mime="application/zip",
                            help=f"Download all masked images in {output_format} format along with the PII data JSON in a single ZIP archive."
                        )
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("⬆️ Upload New Files / Start Over", on_click=self.reset_upload, key="back_to_upload_button")


if __name__ == "__main__":
    app = StreamlitApp()

