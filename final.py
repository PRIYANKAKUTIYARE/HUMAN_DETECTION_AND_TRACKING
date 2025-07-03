
#  STEP 1: SETUP

from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics deep-sort-realtime opencv-python-headless -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


#  STEP 2: PATHS

QUERY_IMAGE_PATH = "/content/drive/MyDrive/HumanTrackingOutput/test/target.png"
TEST_VIDEO_PATH = "/content/drive/MyDrive/HumanTrackingOutput/test/sample1.mp4"
OUTPUT_VIDEO_PATH = "/content/output_test.mp4"
DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/HumanTrackingOutput/output_test.mp4"

# # ================================
# # ✅ STEP 3: IMAGE ENHANCEMENT
# # ================================
# def enhance_image(img):
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(3.0, (8,8))
#     cl = clahe.apply(l)
#     return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def enhance_image(img):
    # Step 1: Convert to LAB and apply CLAHE to the L channel (brightness)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Step 2: Apply Gamma Correction for better brightness boost
    gamma = 1.5  # Try values between 1.2 and 2.0
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_clahe, table)

    return img_gamma

# ================================
# ✅ STEP 4: GET QUERY EMBEDDING
# ================================
def get_query_embedding(image_path, yolo_model, deepsort):
    img = cv2.imread(image_path)
    img = enhance_image(img)
    results = yolo_model(img)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                emb = deepsort.embedder.predict([crop])[0]
                plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                plt.title("Query Person")
                plt.axis('off')
                plt.show()
                return emb
    raise ValueError("❌ No valid person in query image")

# ================================
# ✅ STEP 5: TRACK & EVALUATE
# ================================
def track_locked_target(video_path, query_emb, yolo_model, deepsort, out_path):
    cap = cv2.VideoCapture(video_path)
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    target_id = None
    correct_detections = 0
    total_frames = 0
    false_positives = 0
    total_detections = 0
    frame_id = 0

    sim_list, id_list, frame_id_list = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        total_frames += 1
        frame = enhance_image(frame)

        results = yolo_model(frame)
        dets, crops = [], []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), 'person'))
                crops.append(crop)

        embeds = deepsort.embedder.predict(crops) if crops else []
        tracks = deepsort.update_tracks(dets, embeds=embeds, frame=frame)

        for i, t in enumerate(tracks):
            if not t.is_confirmed() or not hasattr(t, 'features') or not t.features:
                continue
            track_emb = t.features[-1]
            sim = cosine_similarity([track_emb], [query_emb])[0][0]

            sim_list.append(sim)
            id_list.append(t.track_id)
            frame_id_list.append(frame_id)

            if target_id is None and sim > 0.6:
                target_id = t.track_id
                print(f"✅ Locked onto target ID {target_id} at frame {frame_id}")

            if t.track_id == target_id:
                total_detections += 1
                if sim > 0.5:
                    correct_detections += 1
                else:
                    false_positives += 1

                x1, y1, x2, y2 = map(int, t.to_ltrb())
                label = f'Target {target_id} ({sim:.2f})'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
        out.write(frame)

    cap.release()
    out.release()
    print("✅ Done. Output saved.")

    # =========================
    # ✅ METRICS CALCULATION
    # =========================
    TP = correct_detections
    FP = false_positives
    FN = total_frames - correct_detections
    TN = 0  # not measurable without full labels

    accuracy = TP / total_frames if total_frames else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n📊 Confusion Matrix Metrics:")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): Not applicable")

    print("\n📈 Classification Report:")
    print(f"✅ Accuracy : {accuracy:.2%}")
    print(f"✅ Precision: {precision:.2%}")
    print(f"✅ Recall   : {recall:.2%}")
    print(f"✅ F1 Score : {f1_score:.2%}")

    # =========================
    # ✅ PLOTS
    # =========================
    df = pd.DataFrame({
        'Frame': frame_id_list,
        'Similarity': sim_list,
        'Track ID': id_list
    })

    # Similarity Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['Frame'], df['Similarity'], label='Similarity')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
    plt.title('Similarity with Query Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Track ID Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['Frame'], df['Track ID'], label='Track ID')
    plt.title('Track ID Over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Track ID')
    plt.grid(True)
    plt.show()

    # Confusion Matrix Plot
    conf_matrix = np.array([[TP, FP],
                            [FN, 0]])  # TN unknown
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted:Yes', 'Predicted:No'],
                yticklabels=['Actual:Yes', 'Actual:No'])
    plt.title("Confusion Matrix (Partial)")
    plt.show()

# ================================
# ✅ STEP 6: RUN EVERYTHING
# ================================
yolo_model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, etc.
deepsort = DeepSort(max_age=90, n_init=2, embedder="mobilenet", half=True)

try:
    query_emb = get_query_embedding(QUERY_IMAGE_PATH, yolo_model, deepsort)
except Exception as e:
    print(e)
    raise SystemExit

track_locked_target(TEST_VIDEO_PATH, query_emb, yolo_model, deepsort, OUTPUT_VIDEO_PATH)
shutil.move(OUTPUT_VIDEO_PATH, DRIVE_OUTPUT_PATH)
print(f"📁 Output video saved to: {DRIVE_OUTPUT_PATH}")
