"""
Collect pose landmark data from exercise videos for training.
------------------------------------------------------------
Input : Folder of videos (e.g. data/pushups/, data/squats/, etc.)
Output: CSV file with pose landmarks and labels.
Each row = one frame's 33 landmarks (x,y) + exercise label.
"""

import cv2
import mediapipe as mp
import csv, os

# ---------- Configuration ----------
# Folder structure:
# data/
#   pushups/
#       vid1.mp4
#       vid2.mp4
#   squats/
#       vid3.mp4
#   pullups/
#       vid4.mp4
#   planks/
#       vid5.mp4
DATA_DIR = "data"
OUTPUT_CSV = "exercise_data.csv"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ---------- Write CSV Header ----------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = []
    for i in range(33):  # 33 landmarks
        header += [f"x{i}", f"y{i}"]
    header.append("label")
    writer.writerow(header)

# ---------- Process all videos ----------
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    print(f"\n📹 Processing exercise: {label}")

    for vid_file in os.listdir(label_dir):
        if not vid_file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        path = os.path.join(label_dir, vid_file)
        cap = cv2.VideoCapture(path)
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue

            # Extract landmark (x, y)
            row = []
            for lm in res.pose_landmarks.landmark:
                row += [lm.x, lm.y]
            row.append(label)

            # Save to CSV
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            if frame_count % 50 == 0:
                print(f"  Processed {frame_count} frames from {vid_file}...")

        cap.release()

print("\n✅ Done! All video landmarks saved to", OUTPUT_CSV)
