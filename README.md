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

- Used OpenCV to draw masks over detected PII coordinates.
- ✅ Final output: Images with sensitive information successfully redacted.

---

## 📊 Evaluation

Manually corrected PII labels on a test set for validation.


- **✅ Accuracy**: 91%  
- **🎯 Precision**: 95%  
- **🔁 Recall**: 95%


## 💻 Tech Stack

- 🧠 **YOLOv8** – ID card detection  
- 🖊️ **LabelStudio** – Manual annotation of bounding boxes  
- 🧾 **Aspose OCR + OpenCV** – Image alignment and deskewing  
- 🔍 **PaddleOCR** – Text extraction and layout analysis  
- 🧠 **Gemini 1.5 Flash** – PII detection  
- 📱 **Streamlit** – User interface for the pipeline  
- 🐍 **Python** – Core implementation  
- 📓 **Jupyter Notebooks** – Prototyping and evaluation  


## 📋 Prerequisites

- Python 3.8+
- Git
- A compatible environment for GPU acceleration (optional, for YOLOv8 inference)

## 🛠️ Setup Instructions

### 🔄 Clone the Repository:

```bash
git clone https://github.com/your-username/sensitive-info-hiding.git
cd sensitive-info-hiding
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## 📥 Download Models:

- Download the trained YOLOv8 model (yolov8_id_card.pt) and place it in the models/ directory.
- Note: Due to file size, the model is not included in the repository. Contact the repository owner or train your own model using the provided scripts.


Prepare Data:

- Place raw images in data/raw/.
- Annotations (if available) should be in data/annotations/ in LabelStudio format.
- ⚠️ Note: Sensitive data is not included in this repository. Users must provide their own dataset.

## 🧪 Usage

### ▶️ Running the Pipeline
Execute the pipeline scripts in sequence:

```bash
python scripts/detect_id_card.py
python scripts/crop_images.py
python scripts/deskew_images.py
python scripts/extract_text.py
python scripts/detect_pii.py
python scripts/mask_pii.py
```

## Running the Streamlit App
- Launch the Streamlit app to interact with the pipeline:
```bash
streamlit run app/app.py
```

The app provides a user-friendly interface to upload images, process them through the pipeline, and view masked outputs.


## 📝 Notes

**Data Privacy**: Do not upload sensitive images or data to GitHub. The data/ directory is a placeholder; users must source their own datasets.
**Model Training**: To train your own YOLOv8 model, use the annotations in data/annotations/ and follow the instructions in scripts/detect_id_card.py.
**Docker Deployment**: The Docker setup is in progress and will be added in a future update.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1.Fork the repository.
2.Create a new branch (git checkout -b feature/your-feature).
3.Commit changes (git commit -m "Add your feature").
4.Push to the branch (git push origin feature/your-feature).
5.Open a pull request.

📬 Contact
For questions or support, please open an issue on GitHub or contact suryavenkata.mds2024@cmi.ac.in
