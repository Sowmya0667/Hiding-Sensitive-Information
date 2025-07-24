import asyncio
import platform
import re
import json
import os
import requests
import time
import random
import base64
from typing import Dict, List, Optional
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configurations
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None
API_CALLS_PER_MINUTE = 4
SECONDS_PER_MINUTE = 60

def encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded
    except Exception as e:
        print(f"Failed to encode image {image_path}, proceeding with text only: {str(e)}")
        return None

def test_gemini_api(max_retries: int = 5) -> bool:
    if not GEMINI_API_KEY:
        print("Gemini API key not found in environment variables. Skipping API test.")
        return False
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hello, this is a test. Respond with 'Test successful'."}]}
        ],
        "generationConfig": {"maxOutputTokens": 10, "temperature": 0.5}
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            success = content == "Test successful"
            print(f"Gemini API test {'successful' if success else 'failed'}")
            return success
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(random.choice([90, 120]))
            else:
                print(f"Gemini API test failed: HTTPError {str(e)}")
                return False
        except requests.RequestException as e:
            print(f"Gemini API test failed: RequestException {str(e)}")
            return False
    print("Gemini API test failed after max retries")
    return False

def is_english_text(text: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9\s.,-/:&]*$', text))

def is_valid_name(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    return bool(re.match(r'^[a-zA-Z\s.&]+$', text))

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
                    if is_valid_name(surname):
                        result["Surname"] = {"value": surname, "coordinates": block.get("coordinates", [])}
                given_name_match = re.search(r'(?:GIVEN NAME\s*[/:])?\s*([A-Z\s]+?)(?=\s+INDIAN|\s+M\s+\d{2}/\d{2}/\d{4}\b|$)', text, re.IGNORECASE)
                if given_name_match:
                    given_names = given_name_match.group(1).strip()
                    if is_valid_name(given_names):
                        result["Given Name(s)"] = {"value": given_names, "coordinates": block.get("coordinates", [])}
            return result

        name_part = mrz_line[5:]
        name_parts = name_part.split('<<')
        if len(name_parts) < 2:
            return result

        surname = name_parts[0].replace('IND', '', 1).replace('<', ' ').strip()
        given_names = ' '.join([part.replace('<', ' ').strip() for part in name_parts[1:] if part.strip()])

        if not is_valid_name(surname) or not is_valid_name(given_names):
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

def replace_null_with_none(data: Dict) -> Dict:
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            result[k] = replace_null_with_none(v)
        return result
    elif isinstance(data, list):
        return [replace_null_with_none(item) for item in data]
    elif data is None:
        return "NONE"
    return data

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

def coordinate_hook(pairs):
    if len(pairs) > 0 and all(key in ['x', 'y'] for key, value in pairs):
        x_count = sum(1 for key, _ in pairs if key == 'x')
        y_count = sum(1 for key, _ in pairs if key == 'y')
        if x_count > 1 or y_count > 1:
            return merge_duplicate_keys(pairs)
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

def merge_duplicate_keys(pairs):
    x_vals = [value for key, value in pairs if key == 'x']
    y_vals = [value for key, value in pairs if key == 'y']
    points = [{'x': x_vals[i], 'y': y_vals[i]} for i in range(min(len(x_vals), len(y_vals)))]
    return points if points else []

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
                    points = merge_duplicate_keys(list(coord_set.items()))
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

def extract_aadhaar_id(json_data: List[Dict], text: str, phone_number: str = "NONE") -> Dict[str, Optional[str]]:
    try:
        aadhaar_id = None
        coordinates = []

        for block in json_data:
            block_text = block.get("text", "")
            block_text_cleaned = re.sub(r'[^0-9]', '', block_text)  
            print(f"Checking text block: '{block_text}' (cleaned: '{block_text_cleaned}')")
            if re.match(r'[2-9]\d{11}$', block_text_cleaned):  
                aadhaar_id = block_text_cleaned
                coords = block.get("coordinates", [])
                if coords and is_valid_coordinate_format([coords]):
                    coordinates.append(coords)
                print(f"Found Aadhaar ID in block: '{block_text}', Aadhaar ID: {aadhaar_id}, Coordinates: {coords}")
                break  

        if not aadhaar_id:
            cleaned_text = re.sub(r'[^0-9]', '', text)
            aadhaar_match = re.search(r'[2-9]\d{11}', cleaned_text)
            if aadhaar_match:
                aadhaar_id = aadhaar_match.group(0)
                print(f"Found Aadhaar ID in full text: {aadhaar_id}")
                
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
                        if coords and is_valid_coordinate_format([coords]):
                            coordinates.append(coords)
                            print(f"Matched Aadhaar ID in block: '{block_text}', Coordinates: {coords}")
            else:
                print(f"No 12-digit Aadhaar ID found in full text: {cleaned_text}")
                return {"value": "NONE", "coordinates": []}

        if not aadhaar_id:
            print("No valid Aadhaar ID found")
            return {"value": "NONE", "coordinates": []}

        if not coordinates:
            print(f"No coordinates found for Aadhaar ID: {aadhaar_id}")

        return {"value": aadhaar_id, "coordinates": fix_coordinates_format(coordinates)}
    except Exception as e:
        print(f"Error extracting Aadhaar ID: {str(e)}")
        return {"value": "NONE", "coordinates": []}

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
            "- **Aadhaar Card**: Extract Aadhaar ID (exactly 12-digit number, e.g., '986709620066'). Do not extract VID (16 digits). Use the image to verify and improve text accuracy if OCR output is unclear. Extract only the 12-digit Aadhaar number (e.g., '986709620066') that is clearly labeled as Aadhaar Number or appears in expected positions. Do NOT extract the following: a) Any 16-digit number (e.g., VID) b) Any 12-digit sequence that is part of a 16-digit number c) Any number unless you are confident it is a standalone 12-digit Aadhaar number based on both the image layout and text annotation"
            "- **Phone Number**: Exactly 10-digit number starting with 6-9. A 12-digit number is Aadhaar ID, not Phone Number. "
            "- **Voter ID Card**: Extract Voter ID (10-char code, e.g., 'RXH0704411'). Don't extract VID number as Voter ID "
            "- **Passport**: Extract Passport ID (8-char, e.g., 'H1591111'). Use 'Place of Birth' for Address. Combine 'Surname' and 'Given Name(s)' into 'Name'. Prioritize MRZ data for Name, Date of Birth, and Passport ID if available. Extract Name from the first MRZ line, Passport ID (positions 1-8), and Date of Birth (positions 14-19, YYMMDD to DD/MM/YYYY) from the second MRZ line. Include coordinates from JSON annotations for MRZ text blocks. "
            "- **Driving License**: Prioritize 'Date of Birth' over 'Issue Date'. Date format: DD/MM/YYYY. Father's Name may appear as 'Son/Daughter/Wife of'. "
            "- **Water Bill**: Extract Bill Number (12-digit, e.g., '788659363282'). For names, include multiple individuals (e.g., 'Amit Kumar & Sunita Devi') in separate fields ('Name', 'Name1', etc.). "
            "- **PAN Card**: Extract PAN ID (10-char, e.g., 'ABCDE1234F'). Name and Father's Name may be unlabeled; assume first two all-caps names before PAN ID or DOB. "
            "- **Non-ID Documents (e.g., Reports)**: Extract names (e.g., author, mentors, teammates) as 'Name', 'Name1', 'Name2', etc. Set ID fields (Aadhaar ID, PAN ID, etc.) to {'value': 'NONE', 'coordinates': []} unless explicitly detected. "
            "- Coordinates must be sourced exclusively from JSON annotations. If coordinates are invalid, return an empty list for that field. "
            f"**Text**: {text_safe}\n\n"
            f"**JSON Annotations**: {json_data_str}\n"
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
            "    \"Document Type\": \"Mixed|Text Only|Aadhaar Card|PAN Card|Passport|Driving License|Voter ID Card|Water Bill\",\n"
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
            "    \"ZIP Code\": {\"value\": null, \"coordinates\": []},\n"
            "    \"Bill Number\": {\"value\": null, \"coordinates\": []}\n"
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
            encoded_image = encode_image(image_path)
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
                
                try:
                    parsed_result = json.loads(raw_response, object_pairs_hook=coordinate_hook)
                    if not isinstance(parsed_result, dict):
                        raise ValueError("Parsed result is not a dictionary")
            
                    for field in parsed_result:
                        if field != "Document Type" and isinstance(parsed_result[field], dict) and "coordinates" in parsed_result[field]:
                            raw_coords = parsed_result[field]["coordinates"]
                            if raw_coords:
                                try:
                                    parsed_result[field]["coordinates"] = fix_coordinates_format(raw_coords)
                                except Exception as e:
                                    print(f"Error fixing coordinates for field {field}: {str(e)}")
                                    parsed_result[field]["coordinates"] = []
                            else:
                                parsed_result[field]["coordinates"] = []
                                
                    valid_fields = [
                        "Document Type", "Name", "Father's Name", "Mother's Name", "Husband's Name", "Phone Number",
                        "Date of Birth", "Aadhaar ID", "PAN ID", "Passport ID", "Driving License ID",
                        "Voter ID", "Address", "ZIP Code", "Bill Number"
                    ]
                    
                    parsed_result = {k: v for k, v in parsed_result.items() if k in valid_fields or (k.startswith("Name") and int(k.replace("Name", "")) <= 10)}
                    parsed_result = replace_null_with_none(parsed_result)
                    parsed_result['_raw_response'] = raw_response
                    return parsed_result
                
                except json.JSONDecodeError as e:
                    print(f"Gemini API response parsing failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying... ({attempt + 1}/{max_retries})")
                        continue
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
                        "Bill Number": {"value": "NONE", "coordinates": []},
                        "_raw_response": raw_response
                    }
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    time.sleep(random.choice([90, 120]))
                    continue
                print(f"Gemini API request failed: HTTPError {str(e)}")
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
                    "Bill Number": {"value": "NONE", "coordinates": []},
                    "_raw_response": None
                }
            except requests.RequestException as e:
                print(f"Gemini API request failed: RequestException {str(e)}")
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
                    "Bill Number": {"value": "NONE", "coordinates": []},
                    "_raw_response": None
                }
        print("Gemini API request failed after max retries")
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
            "Bill Number": {"value": "NONE", "coordinates": []},
            "_raw_response": None
        }
    except Exception as e:
        print(f"Error in extract_pii_with_gemini: {str(e)}")
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
            "Bill Number": {"value": "NONE", "coordinates": []},
            "_raw_response": None
        }

def validate_file_pair(json_path: str, image_path: Optional[str]) -> bool:
    try:
        if not image_path:
            return True
        json_name = os.path.basename(json_path).lower()
        image_name = os.path.basename(image_path).lower()
        doc_types = ['aadhaar', 'pan', 'voter', 'passport', 'driving', 'water']
        json_doc_type = next((dt for dt in doc_types if dt in json_name), None)
        image_doc_type = next((dt for dt in doc_types if dt in image_name), None)
        return not (json_doc_type and image_doc_type and json_doc_type != image_doc_type)
    except Exception as e:
        print(f"Error in validate_file_pair: {str(e)}")
        return False

def process_json_file(json_file_path: str, image_path: str = None, use_gemini: bool = True) -> Dict:
    try:
        print(f"Processing JSON file: {json_file_path}")
        if image_path:
            print(f"Processing image: {image_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to load JSON file: {json_file_path}, error: {str(e)}")
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
        
        text = " ".join([block['text'] for block in json_data])
        print(f"Input Text: {text}")
        
        if use_gemini and image_path:
            print("Attempting PII extraction with image...")
            result = extract_pii_with_gemini(text, json_data, image_path)
            if isinstance(result, dict):
                raw_response_with_image = result.get('_raw_response', None)
                print(f"Gemini API raw response (with image): {raw_response_with_image}")
                
                all_valid = True
                for field in result:
                    if field != '_raw_response' and field != "Document Type" and result[field]["coordinates"]:
                        raw_coords = result[field]["coordinates"]
                        try:
                            result[field]["coordinates"] = fix_coordinates_format(raw_coords)
                            if not is_valid_coordinate_format(result[field]["coordinates"]):
                                all_valid = False
                        except Exception as e:
                            print(f"Error fixing coordinates for field {field}: {str(e)}")
                            result[field]["coordinates"] = []
                
                if all_valid:
                    print("All coordinates are in correct format, proceeding with image-based result.")
                    result.pop('_raw_response', None)
                    if result["Document Type"] == "Aadhaar Card" and result["Aadhaar ID"]["value"] == "NONE":
                        aadhaar_result = extract_aadhaar_id(json_data, text, result["Phone Number"]["value"])
                        result["Aadhaar ID"] = aadhaar_result
                        if aadhaar_result["value"] != "NONE" and result["Phone Number"]["value"] == aadhaar_result["value"]:
                            result["Phone Number"] = {"value": "NONE", "coordinates": []}
                    print(f"Gemini PII: {json.dumps({key: result[key].get('value') if key != 'Document Type' else result[key] for key in result}, indent=2, ensure_ascii=False)}")
                    return result
                
                print("Some coordinates are invalid, reprocessing without image but retaining PII values...")
                image_result = result.copy()
                result = extract_pii_with_gemini(text, json_data, None)
                raw_response_without_image = result.get('_raw_response', None) if isinstance(result, dict) else None
                if raw_response_without_image:
                    print(f"Gemini API raw response (without image): {raw_response_without_image}")
                
                for field in result:
                    if field != '_raw_response' and field != "Document Type":
                        result[field]["value"] = image_result[field]["value"]
                        coordinates = []
                        field_value = result[field]["value"].lower() if result[field]["value"] not in ["NONE", None] else ""
                        if field_value:
                            for block in json_data:
                                block_text = block.get("text", "").lower()
                                if field_value in block_text or any(part in block_text for part in field_value.split()):
                                    coords = block.get("coordinates", [])
                                    if coords and is_valid_coordinate_format([coords]):
                                        coordinates.append(coords)
                            result[field]["coordinates"] = fix_coordinates_format(coordinates)
                
                result.pop('_raw_response', None)
                if result["Document Type"] == "Aadhaar Card" and result["Aadhaar ID"]["value"] == "NONE":
                    aadhaar_result = extract_aadhaar_id(json_data, text, result["Phone Number"]["value"])
                    result["Aadhaar ID"] = aadhaar_result
                    if aadhaar_result["value"] != "NONE" and result["Phone Number"]["value"] == aadhaar_result["value"]:
                        result["Phone Number"] = {"value": "NONE", "coordinates": []}
                print(f"Gemini PII: {json.dumps({key: result[key].get('value') if key != 'Document Type' else result[key] for key in result}, indent=2, ensure_ascii=False)}")
                return result
            else:
                print("Invalid result from Gemini API with image, reprocessing without image...")
                result = extract_pii_with_gemini(text, json_data, None)
                raw_response_without_image = result.get('_raw_response', None) if isinstance(result, dict) else None
                if raw_response_without_image:
                    print(f"Gemini API raw response (without image): {raw_response_without_image}")
        elif use_gemini:
            print("Attempting PII extraction without image...")
            result = extract_pii_with_gemini(text, json_data, None)
            raw_response_without_image = result.get('_raw_response', None) if isinstance(result, dict) else None
            if raw_response_without_image:
                print(f"Gemini API raw response (without image): {raw_response_without_image}")
        else:
            result = {
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
        
        if not isinstance(result, dict):
            print(f"Error: Result is not a dictionary, got {type(result)}. Using default result.")
            result = {
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
                result[field] = {"value": name, "coordinates": fix_coordinates_format([coords]) if coords else []}
        
        if result["Document Type"] == "Aadhaar Card" and result["Aadhaar ID"]["value"] == "NONE":
            aadhaar_result = extract_aadhaar_id(json_data, text, result["Phone Number"]["value"])
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

def detect_pii(json_folder: str, image_folder: str, output_folder: str, use_gemini: bool = True):
    """
    Detect PII from JSON annotations and aligned images using Gemini API or regex-based fallback.

    Args:
        json_folder (str): Path to folder with JSON annotations from PaddleOCR.
        image_folder (str): Path to folder with aligned images.
        output_folder (str): Path to save PII JSON files.
        use_gemini (bool): Whether to use Gemini API for PII extraction.
    """
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get list of JSON files
        json_file_paths = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.lower().endswith('.json')]
        json_file_paths.sort()  # Sort for consistent processing order

        # Generate corresponding image and output paths
        image_file_paths = []
        output_file_paths = []

        for json_path in json_file_paths:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            image_path = os.path.join(image_folder, f"{base_name}.jpg")
            if os.path.exists(image_path) and validate_file_pair(json_path, image_path):
                image_file_paths.append(image_path)
            else:
                image_file_paths.append(None)
                print(f"Image not found or invalid pair for {json_path}: {image_path}")
            
            output_path = os.path.join(output_folder, f"{base_name}.json")
            output_file_paths.append(output_path)

        print(f"Found {len(json_file_paths)} JSON files to process.")
        
        # Test Gemini API and process files
        use_gemini = use_gemini and test_gemini_api() and GEMINI_API_KEY is not None
        if not use_gemini:
            print("Gemini API not available or disabled. Using regex-based fallback for PII extraction.")

        total_files = len(json_file_paths)
        processed_count = 0
        gemini_count = 0
        batch_start_time = time.time()

        for idx, (json_file_path, image_path, output_path) in enumerate(zip(json_file_paths, image_file_paths, output_file_paths), 1):
            print(f"\nProcessing file {idx}/{total_files}: {json_file_path}")
            if not os.path.exists(json_file_path):
                print(f"JSON file does not exist: {json_file_path}")
                continue
            if not json_file_path.lower().endswith('.json'):
                print(f"Invalid file type, expected JSON: {json_file_path}")
                continue
            if image_path and not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                image_path = None
            elif image_path and not validate_file_pair(json_file_path, image_path):
                print(f"File pair validation failed for JSON: {json_file_path}, Image: {image_path}")
                continue

            if use_gemini and gemini_count >= API_CALLS_PER_MINUTE:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time < SECONDS_PER_MINUTE:
                    time.sleep(SECONDS_PER_MINUTE - elapsed_time)
                batch_start_time = time.time()
                gemini_count = 0

            try:
                result = process_json_file(json_file_path, image_path, use_gemini)
                print(f"Output to be saved in {output_path}:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {json_file_path}: {str(e)}")
                result = {
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
                print(f"Output to be saved in {output_path}:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Output saved to: {output_path} (error case)")
                continue

            processed_count += 1
            if use_gemini:
                gemini_count += 1
            print(f"Completed processing file {idx}/{total_files}")

    except Exception as e:
        print(f"Error in detect_pii: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Detect PII from JSON annotations and aligned images using Gemini API.")
    parser.add_argument("--json_folder", default="data/processed/paddle_result/annotations", help="Path to folder with JSON annotations")
    parser.add_argument("--image_folder", default="data/processed/aligned", help="Path to folder with aligned images")
    parser.add_argument("--output_folder", default="data/processed/pii", help="Path to save PII JSON files")
    parser.add_argument("--no_gemini", action="store_true", help="Disable Gemini API and use regex-based fallback")
    args = parser.parse_args()

    detect_pii(args.json_folder, args.image_folder, args.output_folder, use_gemini=not args.no_gemini)

async def main_async():
    main()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main_async())
else:
    if __name__ == "__main__":
        main()
