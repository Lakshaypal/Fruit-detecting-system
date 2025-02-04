import os
from ultralytics import YOLO  # Using Ultralytics YOLOv5

# Load YOLO model
model = YOLO("models/best.pt")

# Folders
image_folder = "data/"
output_folder = "results/"
os.makedirs(output_folder, exist_ok=True)

# Process all images
for img_name in sorted(os.listdir(image_folder)):
    if img_name.endswith(".jpg"):
        img_path = os.path.join(image_folder, img_name)
        results = model(img_path)  # Run YOLO detection
        
        # Save results properly
        for i, result in enumerate(results):
            output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_result_{i}.jpg")
            result.save(output_path)
        
        print(f"✅ Processed: {img_name}")

print("\n🎉 All images processed successfully! Results saved in 'results/' folder.")