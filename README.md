# 🔐 Hiding Sensitive Information from Images and Documents
This project implements a six-stage pipeline to detect and hide personally identifiable information (PII) from real-world images of ID cards and other documents, captured under diverse conditions (different angles, backgrounds, and lighting). It combines computer vision, OCR, and large language models to achieve high accuracy in PII detection and masking. A Streamlit app is included for user-friendly interaction with the pipeline.

---

## 🚀 Pipeline Overview

### ✅ Stage 1: Data Collection

- Collected real-world document images from Google and various websites.
- Images vary in background, orientation, and lighting conditions.

### 🧭 Stage 2: ID Card Detection and Cropping

- Annotated images using LabelStudio to create bounding boxes for ID cards.
- Trained a YOLOv8 model on the annotated dataset.
- Detected and cropped ID cards from full document images using predicted bounding boxes.

### 🔄 Stage 3: Image Alignment (Deskewing)

- Used Aspose OCR to calculate the skew angle of cropped images.
- Applied OpenCV to deskew (rotate) images for proper alignment.
- 📈 Result: Well-aligned ID cards optimized for accurate text extraction.

### 📝 Stage 4: OCR Text Extraction

- Utilized PaddleOCR to extract text and obtain layout information (text + coordinates).
- Provides input for LLM-based PII identification.

### 🔎 Stage 5: PII Extraction with LLM

- Employed Gemini 1.5 Flash LLM to detect PII (e.g., name, DOB, ID number).
- Used coordinates from PaddleOCR, as the LLM cannot extract coordinates directly.
- Enhanced noisy OCR text using the LLM for improved PII detection accuracy.

### 🛡️ Stage 6: Sensitive Information Masking

- Used Pillow to draw masks over detected PII coordinates.
- ✅ Final output: Images with sensitive information successfully redacted.

---

## 📊 Evaluation

- Evaluated on a test set with manually corrected PII labels (ground truth created by human annotators).
- **Accuracy**: 91%  
- **Precision**: 95%  
- **Recall**: 95%  
- Metrics reflect the performance of the entire pipeline, from ID card detection to PII masking.

---


## 💻 Tools Used

- 🧠 **YOLOv8** – ID card detection  
- 🖊️ **LabelStudio** – Manual annotation of bounding boxes  
- 🧾 **Aspose OCR + OpenCV** – Image alignment and deskewing  
- 🔍 **PaddleOCR** – Text extraction and layout analysis  
- 🧠 **Gemini 1.5 Flash** – PII detection
- 📓 **Pillow**: PII masking 
- 📱 **Streamlit** – User interface for the pipeline  
- 🐳 **Docker** – Containerized deployment
- 🐍 **Python** – Core implementation






## 📁 Folder Structure

```markdown
Hiding-Sensitive-Information/
├── data/                      # Data configuration and annotations (no images included)
|   ├── raw/                   # Some raw images
│   ├── annotations/           # LabelStudio annotations for training
│   ├── data.yaml              # YOLOv8 data configuration file
│   └── processed/             # Temporary pipeline outputs 
│       ├── cropped/           # Cropped ID card images
│       ├── aligned/           # Deskewed images
│       └── masked/            # Masked images
├── models/                    # Trained models
│   └── README.md              # YOLOv8 model weights
├── scripts/                   # Core pipeline scripts
│   ├── train_yolov8.py        # Train YOLOv8 model
│   ├── predict_yolov8.py      # Predict ID card bounding boxes
│   ├── crop_images.py         # Crop ID cards
│   ├── deskew_images.py       # Deskew images
│   ├── extract_text.py        # PaddleOCR text extraction
│   ├── detect_pii.py          # PII detection with Gemini LLM
│   ├── mask_pii.py            # Mask PII with Pillow
│   └── evaluate.py            # Evaluate accuracy, precision, recall
├── app/                       # Streamlit app
│   ├── app.py                 # Streamlit app code
├── Report/                    # Project report
│   └── Sowmya_Summer_Internship_Report.pdf
├── Docker                     # Docker Deployment of the app
│   ├── Dockerfile             # Dockerfile for building the app
|   └── docker_deployment.pdf  # Docker deployment documentation
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and instructions
├── LICENSE                    # Project license
└── .gitignore                 # Git ignore rules

```

---
## 📋 Prerequisites

- Python 3.8+
- Git
- Docker (for containerized deployment)
- A compatible environment for GPU acceleration (optional, for YOLOv8 training and inference)
- API access for Gemini 1.5 Flash (requires configuration)
- PaddleOCR and Aspose OCR dependencies (see requirements.txt)
- Additional dependencies for evaluation: fuzzywuzzy, numpy, python-Levenshtein (optional, for faster fuzzy matching)
---

## 🛠️ Setup Instructions

### 🔄 Clone the Repository:

```bash
git clone https://github.com/Sowmya0667/Hiding-Sensitive-Information.git
cd Hiding-Sensitive-Information
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 📥 Download Models:

- Download the trained YOLOv8 model (yolov8_id_card.pt) and place it in the models/ directory.


## 🐳 Docker Deployment

The PII Masking App can be run locally using Docker for a containerized environment. Follow these steps to deploy the app using the pre-built Docker image.

### Prerequisites
- **Docker Installed**: Ensure **Docker Desktop** is installed and running on your machine. Visit Docker's official website for installation instructions.
- **GEMINI_API_KEY**: Obtain a valid API key for the **Gemini 1.5 Flash LLM**, required for the app to function.

### Deploying and Using the App

**1. Pull the Pre-Built Docker Image:**
- Open a terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux).
- Pull the pii_masking_app_v2 image from Docker Hub using:

```bash
docker pull bodasowmya235/pii_masking_app_v2
```

- This downloads the latest version of the app image to your local machine.

**2. Launch the Docker Container:**
- Run the container, mapping port 8502 on your local machine to port 8502 in the container, and set the GEMINI_API_KEY environment variable. If port 8502 is occupied, use an alternative local port (e.g., -p 8503:8502).

```bash
docker run -p 8502:8502 -e GEMINI_API_KEY=<YOUR_KEY> bodasowmya235/pii_masking_app_v2
```

**3.Interact with the App:**
- Open a web browser and navigate to http://localhost:8502 to access the Streamlit app.
- **Using the App:**
  - **Upload Images**: Click the file upload button to select images of ID cards or documents containing PII (e.g., names, dates of birth, ID numbers).
  - **Process Images**: The app automatically runs the pipeline, detecting and masking PII using the trained YOLOv8 model, PaddleOCR, and Gemini 1.5 Flash LLM.
  - **View Results**: Preview the masked images with redacted PII directly in the app. Download the processed images if needed.
  - **Tips:** Ensure images are clear and well-lit for optimal PII detection. Supported formats include JPG, JPEG, PNG, BMP and PDF.

## Prepare Data

- Place raw images in data/raw/ or test images in data/test/.
- Place the YOLOv8 data configuration file (data.yaml) in data/.
- Annotations should be in data/annotations/ in LabelStudio format.
- ⚠️ **Warning**: Sensitive data is not included in this repository. Users must provide their own dataset. Ensure sensitive files are excluded from version control using .gitignore.

## 🧪 Usage

### ▶️ Running the Pipeline
Execute the pipeline scripts in sequence:

```bash
python scripts/train_yolov8.py --data_path data/data.yaml
python scripts/predict_yolov8.py --model_path models/yolov8_id_card.pt --source data/test --output_dir data/predictions
python scripts/crop_images.py --image_folder data/test --label_folder data/predictions/predict_train/labels --crop_folder data/processed/cropped
python scripts/deskew_images.py --input_folder data/processed/cropped --output_folder data/processed/aligned
python scripts/extract_text.py --input_folder data/processed/aligned --output_image_folder data/processed/paddle_result/images --output_json_folder data/processed/paddle_result/annotations
python scripts/detect_pii.py --json_folder data/processed/paddle_result/annotations --image_folder data/processed/aligned --output_folder data/processed/pii
python scripts/mask_pii.py --json_folder data/processed/pii --image_folder data/processed/aligned --output_folder data/processed/masked
```

## Running the Streamlit App
- Launch the Streamlit app to upload images, process them through the pipeline, and view masked outputs:
```bash
streamlit run app/app.py
```

The app provides a user-friendly interface to upload images, process them through the pipeline, and view masked outputs.


## 📝 Notes

- **Data Privacy**: Do not upload sensitive images or data to GitHub. The data/ directory is a placeholder; users must source their own datasets.
- **Model Training**: Ensure data.yaml points to your dataset in data/annotations/ and data/raw/. Follow instructions in scripts/train_yolov8.py.
- **Docker Deployment**: The app is deployed in a Docker container for portability. Refer to the Docker Deployment section for running the app locally.


 ## 📄 License

This project is licensed under the terms specified in the LICENSE file.
