
# ✅ STEP 1: INSTALL DEPENDENCIES

!pip install -q ultralytics deep_sort_realtime opencv-python-headless scikit-learn tqdm

# ✅ STEP 2: IMPORT LIBRARIES

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import numpy as np
from time import perf_counter
from google.colab import drive
from tqdm import tqdm


# ✅ STEP 3: MOUNT GOOGLE DRIVE

drive.mount('/content/drive')

# ===============================================
# ✅ STEP 4: SETUP PATHS
# ===============================================
ROOT = "/content/drive/MyDrive/human_detection_tracking_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")

train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".mp4")]
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".mp4")]

if not test_files:
    raise FileNotFoundError("❌ No .mp4 files found in test folder.")

# ===============================================
# ✅ STEP 5: VIDEO WRITER FUNCTION
# ===============================================
def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))

# ===============================================
# ✅ STEP 6: TRACKING FUNCTION WITH HIGH PRECISION
# ===============================================
def track_and_evaluate(video_path, save_path, model, tracker):
    cap = cv2.VideoCapture(video_path)
    out = video_writer_same_codec(cap, save_path)

    preds, truths = [], []
    frame_count, total_time = 0, 0

    while True:
        start = perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.4, iou=0.45, classes=0, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()
        formatted = []
        gt = 0

        for *xyxy, conf, cls in detections:
            if conf < 0.4: continue  # Strict filtering
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            formatted.append(([x1, y1, w, h], float(conf), "person"))
            gt += 1

        tracks = tracker.update_tracks(formatted, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        preds.append(len(formatted))
        truths.append(gt)

        out.write(frame)
        frame_count += 1
        total_time += perf_counter() - start

    cap.release()
    out.release()

    preds_np, truths_np = np.array(preds), np.array(truths)
    precision = np.mean(np.where(preds_np > 0, np.minimum(preds_np, truths_np) / preds_np, 0))
    accuracy = np.mean(np.where(truths_np > 0, np.minimum(preds_np, truths_np) / truths_np, 0))

    print(f"✅ {os.path.basename(video_path)}: {frame_count} frames processed.")
    print(f"⚡ Avg FPS: {frame_count / total_time:.2f}")
    print(f"🎯 Precision: {precision * 100:.2f}%")
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

    return precision, accuracy

# ===============================================
# ✅ STEP 7: INIT YOLOv8 (HIGH ACCURACY) + DeepSORT
# ===============================================
model = YOLO("yolov8m.pt")  # 👈 More accurate than yolov8n/s
tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=1.0)

# ===============================================
# ✅ STEP 8: RUN EVALUATION ON TEST SET
# ===============================================
total_p, total_a = 0, 0
for test_vid in tqdm(test_files):
    in_path = os.path.join(TEST_DIR, test_vid)
    out_path = os.path.join(TEST_DIR, f"tracked_{test_vid}")
    p, a = track_and_evaluate(in_path, out_path, model, tracker)
    total_p += p
    total_a += a

print("\n📊 Final Evaluation on Test Set:")
print(f"✅ Avg Precision: {(total_p / len(test_files)) * 100:.2f}%")
print(f"✅ Avg Accuracy: {(total_a / len(test_files)) * 100:.2f}%")
