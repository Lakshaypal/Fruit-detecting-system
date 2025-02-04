# 🍎 UAS-DTU Fruit Counting Task
This project uses **YOLO (You Only Look Once)** for **fruit detection and counting** from **front and back images** of plants.

## 🚀 How to Run the Project?
### 1️⃣ Install Dependencies
Before running the scripts, install the required Python libraries:  
```bash
pip install ultralytics opencv-python numpy
```

### 2️⃣ Run YOLO Detection on Images
Detect fruits in all images and **save results** in the `results/` folder:  
```bash
python scripts/detect.py
```
👉 This will process **all images in `data/`** and save the **detected images in `results/`**.

### 3️⃣ Count Unique Fruits Using Front & Back Views
To **count fruits uniquely** (avoiding duplicate detections from both sides), run:  
```bash
python scripts/process_all.py
```
👉 This will **count unique fruits** based on bounding box positions and print the total fruit count.

## 💡 Explanation of Scripts
### `detect.py` (Object Detection)
- Loads **YOLO model (`best.pt`)**  
- Runs YOLO detection on all **images in `data/`**  
- Saves **results in `results/`**  

### `process_all.py` (Fruit Counting)
- **Extracts fruit detections** from YOLO's bounding boxes  
- **Matches front & back detections** to **avoid duplicates**  
- **Prints total unique fruit count**  
