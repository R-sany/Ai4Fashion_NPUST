import ultralytics.utils.checks



from IPython import display
from ultralytics.models import yolo
from ultralytics.utils import checks
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from roboflow.core import project
import os

from ultralytics import YOLO

def train_model():
    model_path = 'yolov8x-seg.pt'
    model = YOLO(model_path)
    data_yaml = r"D:\final_model_version2\data.yaml"                  # Train 33
    model.train(data=data_yaml, epochs=50, batch=16, imgsz=640, device='cuda')
    image_path = "D:\\test folder\\pic5.jpg"
    results = model.predict(source='D:\\test folder\\pic5.jpg', conf=0.4)
    display_results(results, image_path)
def display_results(results, image_path):
    # Check if results is a list (e.g., for batch processing)
    if isinstance(results, list):
        results = results[0]  # Use the first result if multiple results are returned

    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Create a figure and axis for matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)

    # Extract predictions
    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2]
    masks = results.masks.cpu().numpy() # Segmentation masks as a list of arrays
    class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
    confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
    class_names = results.names  # Class names from model

    # Draw bounding boxes and segmentation masks
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        confidence = confidences[i]
        class_id = class_ids[i]
        label = class_names[class_id]

        # Draw rectangle
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Draw label
        label_text = f"{label} {confidence:.2f}"
        ax.text(x1, y1 - 10, label_text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Draw segmentation masks (if available)
    if masks is not None:
        for mask in masks:
            mask_np = bool(mask.cpu().numpy())  # Convert mask to NumPy array and boolean type
            masked_img = img_rgb.copy()
            masked_img[~mask_np] = 0  # Apply mask to image
            ax.imshow(masked_img, alpha=0.5)  # Overlay mask with transparency

    plt.axis('off')  # Hide the axes
    plt.show()       # Show the plot



if __name__ == '__main__':
    train_model()