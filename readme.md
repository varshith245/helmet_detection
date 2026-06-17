# YOLOv8 Helmet Detection

Real-time helmet detection using YOLOv8 for industrial safety applications. This project provides an end-to-end pipeline from dataset collection and annotation to training, inference, REST API deployment, and Docker containerization.

## Features
- **YOLOv8 Model**: Accurate real-time helmet detection.
- **REST API**: Built with FastAPI to upload images and receive bounding boxes, labels, and an annotated image.
- **Docker Support**: Containerized for seamless deployment.
- **Modular Scripts**: Scripts and notebooks for training, validation, and inference.

---

## Project Structure
```text
helmet_detection/
├── app/
│   ├── app.py           # FastAPI application
├── data/
│   ├── images/          # Train & Validation images
│   └── labels/          # YOLO format labels (.txt)
├── model/
│   └── yolov8/          # Saved model weights (.pt files)
├── notebooks/
│   └── helmet_detection_yolov8.ipynb
├── src/
│   ├── train.py
│   ├── detect.py
│   └── utils.py
├── .env
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/varshith245/helmet_detection.git
cd helmet_detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Model Training
You can train the model either via Jupyter Notebook or using the provided modular scripts.

**Via Notebook:**
1. Open `notebooks/helmet_detection_yolov8.ipynb`.
2. Configure your dataset paths in `data.yaml`.
3. Run the cells to train the YOLOv8 model.

**Via Script:**
```bash
python src/train.py --data data/data.yaml --epochs 50 --img-size 640 --batch-size 16
```

### REST API (FastAPI)

1. Check `app/app.py` and ensure the `MODEL_PATH` variable points to your trained YOLOv8 model weights (e.g., `best.pt`).
2. Start the FastAPI server:
   ```bash
   uvicorn app.app:app --reload
   ```
3. By default, the API will be hosted at `http://127.0.0.1:8000/`.
4. Swagger UI Documentation is available at `http://127.0.0.1:8000/docs`.

### API Endpoints
- `GET /` : Health check.
- `POST /predict/` : Upload an image to detect helmets. Returns a JSON with detections and a URL to download the annotated image.
- `GET /download/{image_name}` : Endpoint to retrieve the dynamically annotated image.

### Testing the API using `curl`
```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict/
```

---

## Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t helmet-detector .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 helmet-detector
   ```

3. Access the API at `http://localhost:8000/` or upload test images via the docs at `http://localhost:8000/docs`.

---

## Future Enhancements
- Helmet-type classification (e.g., color, safety tier).
- Multi-camera integration & dashboard for real-time video stream processing.
- Sound alerts and notifications for safety violations.
