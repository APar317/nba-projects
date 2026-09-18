# Oral Squamous Cell Carcinoma (OSCC) Detector

Deep-learning-powered histopathological biopsy image classification application using **EfficientNetB3** and **Flask**.

Diagnoses histopathology slides into **Normal** tissue or **OSCC (Oral Squamous Cell Carcinoma)** with high confidence (97%+ accuracy on reference OSCC biopsy slides).

---

## Quick Start: Running Locally (Python)

### 1. Prerequisites
- Python 3.10
- Virtual environment (`venv`)

### 2. Activate Environment & Run
```bash
# Navigate to the project folder
cd OralCancer

# Activate virtual environment
source venv/bin/activate

# (Optional) Install dependencies if running in a new environment
pip install -r requirements.txt

# Start the Flask development server
python app.py
```

Open your browser at:  
👉 **http://localhost:5001** (or **http://127.0.0.1:5001**)

---

## Docker Commands

The project includes a self-contained Dockerfile configured for production with **Gunicorn** on port **5001**. All model weights and sample galleries are packaged directly inside the container.

### 1. Build the Docker Image
```bash
docker build -t oral-cancer-detector:latest .
```

### 2. Run the Container in Background
```bash
docker run -d --name oral-cancer-server -p 5001:5001 oral-cancer-detector:latest
```
Access the application at: **http://localhost:5001**

### 3. Stream Live Server Logs
```bash
docker logs -f oral-cancer-server
```
*(Press `Ctrl + C` to exit log streaming; the container remains running in the background).*

### 4. Check Container Status
```bash
docker ps
```

### 5. Validate Inference Inside the Container
Run a direct model prediction check on an OSCC sample image:
```bash
docker exec -it oral-cancer-server python -c "from main import getPrediction; print(getPrediction('static/samples/oscc/OSCC_100x_101.jpg'))"
```
*Expected Output:*
```json
{"label": "OSCC", "confidence": 97.02}
```

### 6. Stop & Remove Container
```bash
# Stop the running container
docker stop oral-cancer-server

# Force-remove the container
docker rm -f oral-cancer-server
```

---

## Developer Handoff / Exporting the Image

To share the pre-built Docker image with other developers or deploy to another machine:

### 1. Export Image to Compressed Archive
```bash
# Export the image to a tar file
docker save -o oral-cancer-detector.tar oral-cancer-detector:latest

# Compress into tar.gz
tar -zcf oral-cancer-detector.tar.gz oral-cancer-detector.tar

# (Optional) Remove the uncompressed tar file to save disk space
rm oral-cancer-detector.tar
```

### 2. How the Developer Loads & Runs the Image
Send `oral-cancer-detector.tar.gz` to the developer. On their machine, they run:
```bash
# Extract and load the image
tar -zxf oral-cancer-detector.tar.gz
docker load -i oral-cancer-detector.tar

# Run the container
docker run -d --name oral-cancer-app -p 5001:5001 oral-cancer-detector:latest
```
The developer can immediately open **http://localhost:5001** with zero configuration.

---

## Project Structure

```text
OralCancer/
├── app.py                     # Flask web application routes & image upload handler
├── main.py                    # EfficientNetB3 architecture, Keras 3 weight loader & inference logic
├── Dockerfile                 # Container definition configured with Gunicorn on port 5001
├── .dockerignore              # Clean build context exclusion list
├── requirements.txt           # Pinned production runtime dependencies
├── models_dump/
│   └── model_3.h5             # Trained OSCC-CNN EfficientNetB3 model weights
├── static/
│   ├── css/styles.css         # Clean UI styling
│   ├── images/                # Uploaded biopsy images directory
│   └── samples/               # Preloaded reference Normal & OSCC histopathology samples
│       └── manifest.json      # Sample metadata & labels
└── templates/
    ├── base.html              # Core navigation layout
    ├── index.html             # Analysis & sample selection portal
    ├── index_report.html      # Diagnostic prediction report card
    └── samples.html           # Reference gallery viewer
```
