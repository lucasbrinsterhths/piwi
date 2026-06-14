import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════
#  1. MASTER CONFIGURATION
# ═══════════════════════════════════════════════════════════
# MODES: 
#  - 'sample_and_label': Randomly picks frames and launches the annotation GUI
#  - 'train': Trains the YOLO model on your saved labels
#  - 'inference': Runs the trained model on video frames
MODE = 'sample_and_label'  

VIDEO_PATH = 'sample/sampleVial.MOV'
FRAMES_DIR = 'sample/frames'
PLOTS_DIR  = 'sample/plots'
DATASET_DIR = 'pupae_dataset'

CROP = (300, 1500, 25, 465)   # y1, y2, x1, x2

# Sampling Configuration
NUM_TRAIN_FRAMES = 15  # Number of random frames to pick for training
NUM_VAL_FRAMES = 3     # Number of random frames to pick for validation

MODEL_PATH = 'runs/detect/train/weights/best.pt' # Switch to 'yolov8n.pt' if not trained yet
CONF_THRESH = 0.35  


# ═══════════════════════════════════════════════════════════
#  2. AUTOMATED DIRECTORY SETUP & SAMPLING
# ═══════════════════════════════════════════════════════════
def setup_dataset_structure():
    subdirs = [
        'dataset/images/train', 'dataset/images/val',
        'dataset/labels/train', 'dataset/labels/val'
    ]
    for folder in subdirs:
        os.makedirs(os.path.join(DATASET_DIR, folder), exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

def sample_random_frames():
    """Extracts random frames ONLY if the dataset folders are currently empty."""
    setup_dataset_structure()
    
    train_img_dir = os.path.join(DATASET_DIR, 'dataset/images/train')
    val_img_dir = os.path.join(DATASET_DIR, 'dataset/images/val')
    
    # Check if we already have frames extracted from a previous session
    if len(os.listdir(train_img_dir)) > 0 or len(os.listdir(val_img_dir)) > 0:
        print("→ Found existing frames in dataset folders! Skipping random sampling to preserve your session.")
        print("→ Launching labeling wizard to catch up on unannotated images...")
        return

    # Otherwise, if it's completely empty, proceed with extracting new random frames
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"CRITICAL: Cannot open video file at {VIDEO_PATH}")
        
    all_frames = []
    y1, y2, x1, x2 = CROP
    
    print("→ Reading video file stream...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        roi = frame[y1:min(y2, h), x1:min(x2, w)]
        all_frames.append(roi)
    cap.release()
    
    total_extracted = len(all_frames)
    if total_extracted == 0:
        raise RuntimeError("No frames could be extracted from the video file.")
        
    print(f"✓ Extracted {total_extracted} total frames from video.")
    
    indices = list(range(total_extracted))
    random.shuffle(indices)
    
    train_idx = indices[:NUM_TRAIN_FRAMES]
    val_idx = indices[NUM_TRAIN_FRAMES:NUM_TRAIN_FRAMES + NUM_VAL_FRAMES]
    
    for idx in train_idx:
        cv2.imwrite(os.path.join(train_img_dir, f'frame_{idx}.jpg'), all_frames[idx])
    for idx in val_idx:
        cv2.imwrite(os.path.join(val_img_dir, f'frame_{idx}.jpg'), all_frames[idx])
        
    print(f"✓ Randomly allocated {len(train_idx)} frames to TRAIN and {len(val_idx)} frames to VAL.")


# ═══════════════════════════════════════════════════════════
#  3. INTERACTIVE MATPLOTLIB LABELING GUI
# ═══════════════════════════════════════════════════════════
class PupaLabelerGUI:
    """A lightweight Matplotlib canvas tool to build bounding boxes via clicks."""
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.img = cv2.imread(image_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]
        
        self.boxes = []      # Format: [[x_center, y_center, width, height], ...] normalized
        self.clicks = []     # Stores active click pairs [(x1, y1), (x2, y2)]
        
        # Setup Interactive Plot Frame
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.img_rgb)
        self.ax.set_title(f"Labeling: {os.path.basename(image_path)}\n"
                          f"Click Top-Left then Bottom-Right of a Pupa.\n"
                          f"Press [d] to Undo | Close Window to Save & Next Frame", 
                          fontsize=11, fontweight='bold')
        
        # Connect event hooks
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        self.clicks.append((event.xdata, event.ydata))
        
        # Plot a temporary guide point where clicked
        self.ax.plot(event.xdata, event.ydata, 'ro', markersize=4)
        self.fig.canvas.draw()
        
        # Once we get a pair of clicks, convert to a YOLO bounding box
        if len(self.clicks) == 2:
            x1, y1 = self.clicks[0]
            x2, y2 = self.clicks[1]
            
            # Reset click buffer
            self.clicks = []
            
            # Compute bounding limits
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            # Normalize to absolute float values between 0.0 and 1.0 for YOLO
            bw = (xmax - xmin) / self.w
            bh = (ymax - ymin) / self.h
            bx = (xmin + (xmax - xmin) / 2.0) / self.w
            by = (ymin + (ymax - ymin) / 2.0) / self.h
            
            self.boxes.append([0, bx, by, bw, bh]) # Class ID 0 for 'pupa'
            self.redraw_canvas()
            
    def on_key(self, event):
        if event.key == 'd' and self.boxes:
            self.boxes.pop()  # Undo last box
            self.redraw_canvas()

    def redraw_canvas(self):
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.ax.set_title(f"Labeling: {os.path.basename(self.image_path)}\nBoxes count: {len(self.boxes)}", fontsize=11, fontweight='bold')
        
        # Draw active bounding containers onto screen
        for box in self.boxes:
            _, bx, by, bw, bh = box
            # Denormalize back to pixel space for visual drawing
            p_cx, p_cy = bx * self.w, by * self.h
            p_w, p_h = bw * self.w, bh * self.h
            
            rect = plt.Rectangle((p_cx - p_w/2, p_cy - p_h/2), p_w, p_h,
                                 fill=False, color='lime', linewidth=2)
            self.ax.add_patch(rect)
        self.fig.canvas.draw()

    def save_yolo_labels(self):
        with open(self.label_path, 'w') as f:
            for box in self.boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def launch_annotation_wizard():
    """Loops through training and validation images to open the interface sequentially."""
    print("\n--- LAUNCHING LABELING INTERFACE ---")
    
    # Process both subsets
    for subset in ['train', 'val']:
        img_dir = os.path.join(DATASET_DIR, f'dataset/images/{subset}')
        lbl_dir = os.path.join(DATASET_DIR, f'dataset/labels/{subset}')
        
        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
            
            # Skip frames already completed
            if os.path.exists(lbl_path):
                continue
                
            # Open GUI loop
            labeler = PupaLabelerGUI(img_path, lbl_path)
            plt.show() # Halts script execution until figure is closed manually
            labeler.save_yolo_labels()
            
    print("✓ Labeling wizard complete! Annotations are saved formatted for training.")


# ═══════════════════════════════════════════════════════════
#  4. MODEL SYSTEM TRAINING & INFERENCE
# ═══════════════════════════════════════════════════════════
def run_training_pipeline():
    print("\n=== STARTING TRAINING MODE ===")
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    yaml_content = f"path: {os.path.abspath(DATASET_DIR)}/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: pupa"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

    train_lbl_dir = os.path.join(DATASET_DIR, 'dataset/labels/train')
    if not os.path.exists(train_lbl_dir) or len(os.listdir(train_lbl_dir)) == 0:
        print("Stopping: Labels missing. Run with MODE = 'sample_and_label' first.")
        return

    model = YOLO('yolov8n.pt')
    model.train(data=yaml_path, epochs=100, imgsz=640, device='cpu')

def run_inference_pipeline():
    print("\n=== STARTING INFERENCE & COUNTING MODE ===")
    model = YOLO(MODEL_PATH)
    
    # Grabs a sample image out of training to prove operation
    test_img = os.path.join(DATASET_DIR, 'dataset/images/train', os.listdir(os.path.join(DATASET_DIR, 'dataset/images/train'))[0])
    img = cv2.imread(test_img)
    
    results = model.predict(source=img, conf=CONF_THRESH)[0]
    pupal_count = len(results.boxes)
    print(f"✓ Absolute Count Total: n = {pupal_count}")
    
    annotated_img = results.plot()
    cv2.putText(annotated_img, f'Total Count n = {pupal_count}', (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)
    
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    if MODE == 'sample_and_label':
        sample_random_frames()
        launch_annotation_wizard()
    elif MODE == 'train':
        run_training_pipeline()
    elif MODE == 'inference':
        run_inference_pipeline()
