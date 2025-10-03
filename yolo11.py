import cv2
import os
from PIL import Image
from ultralytics import YOLO
from fashion_clip.fashion_clip import FashionCLIP
from fashionclip import fclip_prediction


# Load models once (outside the function so they don’t reload each time)
yolo_model = YOLO("yolov8n.pt")   # Replace with your trained model
fclip = FashionCLIP("fashion-clip")


# Define possible categories and colors
CATEGORIES = [
    "flare pants", "straight-leg trousers", "skinny jeans", "blazer", "t-shirt", 
    "tank top", "sneakers", "heels", "loafers", "ankle boots", "sandal", "handbag", 
    "crossbody bag", "backpack", "belt bag", "watch", "bracelet", "necklace", "earrings", 
    "ring", "sunglasses", "hat", "cap", "beanie", "scarf", "gloves", "hoodie", "suit", 
    "pantsuit", "trench coat", "a person carrying a handbag", 
    "shirt", "blouse", "dress", "jacket", "coat", "skirt", "shorts", "sweater"
]

COLORS = [
    "red", "blue", "black", "white", "green", "yellow", "beige", 
    "brown", "grey", "pink", "purple", "orange"
]


def analyze_outfit(image_path, conf_thresh=0.4, save_output=True):
    """
    Detects fashion items in an image and labels them with fine-grained category + color.
    
    Args:
        image_path (str): Path to input image.
        conf_thresh (float): YOLO confidence threshold.
        save_output (bool): Whether to save an annotated output image.
        
    Returns:
        results (list): List of detections with bounding boxes and labels.
    """
    # Run YOLOv8 detection
    results = yolo_model(image_path, conf=conf_thresh)

    # Load image for drawing
    img = cv2.imread(image_path)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)

            # Crop region
            crop = Image.open(image_path).crop((x1, y1, x2, y2)).convert("RGB")

            # FashionCLIP predictions
            pred_cat = fclip_prediction([crop], CATEGORIES, fclip)
            pred_color = fclip_prediction([crop], COLORS, fclip)

            label = f"{pred_color[0]} {pred_cat[0]}"

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(score),
                "category": pred_cat[0],
                "color": pred_color[0],
                "label": label
            })

            # Draw on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if save_output:
        output_path = image_path.replace(".png", "_out.png").replace(".jpg", "_out.jpg")
        cv2.imwrite(output_path, img)

    return detections


if __name__ == "__main__":
    folder_path = "/Users/tracieluong/Documents/runway-images/ralph-lauren/spring-2026-ready-to-wear"  # Replace with your folder path
    supported_exts = [".jpg", ".jpeg", ".png"]  # Add more if needed

    # Get list of image files in folder
    image_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in supported_exts]

    all_results = {}

    for image_path in image_files:
        print(f"Processing {image_path}...")
        detections = analyze_outfit(image_path)
        all_results[image_path] = detections

    # Print results
    for img_path, dets in all_results.items():
        print(f"\nResults for {img_path}:")
        for det in dets:
            print(det)