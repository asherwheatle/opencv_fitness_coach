import cv2
import numpy as np
import mediapipe as mp
import time, threading
from collections import deque
import pyttsx3
import matplotlib.pyplot as plt

# ---------- TEXT TO SPEECH ----------
engine = pyttsx3.init()
engine.setProperty('rate', 185)
engine.setProperty('volume', 1.0)

def speak_async(text):
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

def draw_text(view, text, pos, color=(10,10,10), scale=0.8, thick=2):
    """Draw outlined text for better visibility."""
    x, y = pos
    # white outline
    cv2.putText(view, text, (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3)
    # main colored text
    cv2.putText(view, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ---------- Pose Setup ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark

# ---------- Config ----------
CFG = {
    "squat": {"min_down": 60, "max_up": 130}  # tuned for depth range
}

# ---------- Utility ----------
def get(lms, idx, w, h):
    lm = lms[idx.value]
    return (lm.x * w, lm.y * h, lm.z * w, lm.visibility)

def ok_vis(*vals, t=0.5):
    return all(v is not None and v >= t for v in vals)

# ---------- Analysis ----------
def analyze_squat(lms, w, h):
    """Return: depth, ratio, hip_y, knee_y."""
    rh = get(lms, L.RIGHT_HIP, w, h)
    lh = get(lms, L.LEFT_HIP,  w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    la = get(lms, L.LEFT_ANKLE,  w, h)
    rk = get(lms, L.RIGHT_KNEE, w, h)
    lk = get(lms, L.LEFT_KNEE,  w, h)
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    ls = get(lms, L.LEFT_SHOULDER,  w, h)

    hip_y = knee_y = depth = ratio = None

    # Depth (hip vs ankle)
    if ok_vis(rh[3], lh[3], ra[3], la[3]):
        hip_y = (rh[1] + lh[1]) / 2
        ankle_y = (ra[1] + la[1]) / 2
        depth = ankle_y - hip_y  # larger => deeper squat

    # Knee level
    if ok_vis(rk[3], lk[3]):
        knee_y = (rk[1] + lk[1]) / 2

    # Stance ratio (knee/shoulder)
    if ok_vis(ls[3], rs[3], lk[3], rk[3]):
        shoulder_dist = np.linalg.norm(np.array(ls[:2]) - np.array(rs[:2]))
        knee_dist = np.linalg.norm(np.array(lk[:2]) - np.array(rk[:2]))
        ratio = (knee_dist / shoulder_dist) if shoulder_dist != 0 else None

    return depth, ratio, hip_y, knee_y

# ---------- Smoothing ----------
SMOOTH = {"squat_depth": deque(maxlen=5)}
def smooth_depth(val):
    if val is None:
        return None
    q = SMOOTH["squat_depth"]
    q.append(float(val))
    return float(np.mean(q))

# ---------- Stance feedback ----------
def stance_feedback(ratio):
    if ratio is None:
        return ""
    if ratio < 0.83:
        return "Bring your knees in"
    elif ratio > 1.40:
        return "Spread your knees more"
    else:
        return "Perfect! Keep going!"

# ---------- Rep Counter (only count good reps) ----------
class SquatCounter:
    def __init__(self, low, high, min_frames=2):
        self.low, self.high, self.min_frames = low, high, min_frames
        self.state = "top"
        self.frames = 0

        self.reps = 0          # total good reps
        self.bad_reps = 0      # total bad attempts

        self.bottom_reached = False
        self.max_hip_y = None
        self.bottom_ratio = None
        self.bottom_feedback = ""

        self.last_rep_feedback = ""
        self.last_feedback_time = 0
        self.last_rep_time = 0

    def update(self, depth, hip_y, knee_y, ratio):
        if depth is None or hip_y is None or knee_y is None:
            return None

        now = time.time()
        below_knee = (hip_y > knee_y)  # y increases downward in image

        if self.state == "top":
            if depth >= self.high:
                self.state = "going_down"
                self.frames = 1
                self.bottom_reached = False
                self.max_hip_y = hip_y
                self.bottom_ratio = None
                self.bottom_feedback = ""

        elif self.state == "going_down":
            self.frames += 1
            if below_knee:
                self.bottom_reached = True
                if self.max_hip_y is None or hip_y >= self.max_hip_y:
                    self.max_hip_y = hip_y
                    self.bottom_ratio = ratio
                    self.bottom_feedback = stance_feedback(ratio)

            if self.bottom_reached and self.frames >= self.min_frames:
                self.state = "bottom"
                self.frames = 0

        elif self.state == "bottom":
            if depth <= self.low:
                self.state = "going_up"
                self.frames = 1

        elif self.state == "going_up":
            self.frames += 1
            if depth <= self.low and self.frames >= self.min_frames:
                if now - self.last_rep_time > 0.5:
                    self.last_rep_time = now

                    # ✅ Count rep ONLY if stance was perfect
                    if self.bottom_feedback and "Perfect" in self.bottom_feedback:
                        self.reps += 1
                        self.last_rep_feedback = "✅ Good rep"
                        speak_async("Good rep")
                    else:
                        self.bad_reps += 1
                        fb = self.bottom_feedback if self.bottom_feedback else "Adjust your stance"
                        self.last_rep_feedback = f"❌ {fb}"
                        speak_async(fb)

                    self.last_feedback_time = now

                self.state = "top"
                self.frames = 0

# ---------- Main ----------
def main():
    print("Keys: s=start set, e=end set (show graph), q=quit")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found.")
        return

    counter = SquatCounter(low=CFG["squat"]["min_down"],
                           high=CFG["squat"]["max_up"], min_frames=2)
    time_log, depth_log, ratio_log = [], [], []

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        start_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            view = frame.copy()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            ratio = None
            if res.pose_landmarks:
                mp_draw.draw_landmarks(view, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                depth_raw, ratio_raw, hip_y, knee_y = analyze_squat(res.pose_landmarks.landmark, w, h)
                depth = smooth_depth(depth_raw)
                ratio = ratio_raw

                if depth is not None:
                    depth_log.append(depth)
                    time_log.append(time.time() - start_time)
                if ratio is not None:
                    ratio_log.append(ratio)

                # Update squat logic
                counter.update(depth, hip_y, knee_y, ratio)

                # Always show live ratio
                if ratio is not None:
                    fb = stance_feedback(ratio)
                    color = (0, 200, 0) if "Perfect" in fb else (0, 0, 255) if "Bring" in fb else (255, 200, 0)
                    draw_text(view, f"Knee/Shoulder Ratio: {ratio:.2f}", (10, 100), color, 0.8, 2)

            # HUD
            draw_text(view, "Exercise: SQUAT", (10, 30), (10,10,10), 0.9, 2)
            draw_text(view, f"Good Reps: {counter.reps}", (10, 65), (30,30,30), 0.85, 2)
            draw_text(view, f"Bad Attempts: {counter.bad_reps}", (10, 90), (30,30,30), 0.8, 2)

            # Display feedback after each rep for 2s
            if time.time() - counter.last_feedback_time < 2 and counter.last_rep_feedback:
                good = counter.last_rep_feedback.startswith("✅")
                draw_text(view, counter.last_rep_feedback, (10, 130),
                          (0,200,0) if good else (0,0,255), 0.8, 2)

            cv2.imshow("AI Squat Form Tracker (Only Counts Perfect Reps)", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                counter.state = "top"
                counter.reps = counter.bad_reps = 0
                counter.bottom_reached = False
                counter.bottom_feedback = ""
                time_log.clear()
                depth_log.clear()
                ratio_log.clear()
                speak_async("Starting new set")
            elif key == ord('e'):
                speak_async("Workout ended. Displaying results.")
                total = counter.reps + counter.bad_reps
                accuracy = (counter.reps / total * 100) if total > 0 else 0
                print("\n===== Session Statistics =====")
                print(f"Good reps: {counter.reps}")
                print(f"Bad attempts: {counter.bad_reps}")
                print(f"Accuracy: {accuracy:.1f}%")
                print("==============================\n")

                if depth_log:
                    plt.figure(figsize=(8,4))
                    plt.plot(time_log, depth_log)
                    plt.title("Squat Depth Over Time")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Hip Depth (pixels)")
                    plt.grid(True)
                    plt.show()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
