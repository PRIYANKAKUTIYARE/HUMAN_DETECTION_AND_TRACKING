# === ADD THIS TO THE END OF YOUR EXISTING SCRIPT ===

import pandas as pd

# Variables to collect stats
frame_stats = []

# MODIFIED track_locked_target to collect metrics
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

            # Log for plots
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

    # Save stats
    accuracy = correct_detections / total_frames if total_frames else 0
    precision = correct_detections / total_detections if total_detections else 0
    recall = correct_detections / total_frames if total_frames else 0

    print(f"\n📊 Metrics:")
    print(f"Frames: {total_frames}")
    print(f"Target Detections: {total_detections}")
    print(f"Correct Detections: {correct_detections}")
    print(f"False Positives: {false_positives}")
    print(f"✅ Accuracy: {accuracy:.2%}")
    print(f"✅ Precision: {precision:.2%}")
    print(f"✅ Recall: {recall:.2%}")

    # Graphs
    plot_df = pd.DataFrame({
        'Frame': frame_id_list,
        'Similarity': sim_list,
        'Track ID': id_list
    })

    plt.figure(figsize=(10,4))
    plt.plot(plot_df['Frame'], plot_df['Similarity'], label='Cosine Similarity')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
    plt.title('Similarity with Query over Frames')
    plt.xlabel('Frame')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(plot_df['Frame'], plot_df['Track ID'], label='Track ID')
    plt.title('Track ID over Time')
    plt.xlabel('Frame')
    plt.ylabel('Track ID')
    plt.grid(True)
    plt.show()



# Confusion matrix calculation
TP = correct_detections                         # Correct match of person
FP = false_positives                            # Wrong match (wrong similarity/ID)
FN = total_frames - correct_detections          # Missed frames
TN = 0  # Not computable without full ground truth

# Metrics
accuracy = TP / total_frames if total_frames else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

# Report Confusion Matrix and Metrics
print("\n📊 Confusion Matrix Metrics:")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): Not applicable without ground truth")

print("\n📈 Classification Report:")
print(f"✅ Accuracy : {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall   : {recall:.2%}")
print(f"✅ F1 Score : {f1_score:.2%}")
