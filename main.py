# form_tracker.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# -----------------------------
# Utilities
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

EXERCISES = ["pushup", "pullup", "squat", "plank"]
exercise_idx = 0  # default: pushup

def angle_3pt(a, b, c):
    """
    Returns the angle (in degrees) at point b given three 2D points a,b,c.
    """
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    denom = np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6
    cosang = np.clip(dot / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    if ang > 180:
        ang = 360 - ang
    return float(ang)

def get_point(landmarks, name, w, h):
    lm = landmarks[name.value]
    return (lm.x * w, lm.y * h, lm.visibility)

def visible(*vis_vals, thresh=0.5):
    return all(v is not None and v >= thresh for v in vis_vals)

# -----------------------------
# State & rep logic
# -----------------------------
class RepCounter:
    """
    Generic top/bottom threshold-based rep counter with debouncing.
    """
    def __init__(self, low_thresh, high_thresh, min_frames=3):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.min_frames = min_frames
        self.state = "top"   # top -> going_down -> bottom -> going_up -> top
        self.frames_in_state = 0
        self.reps = 0

    def update_value(self, metric):
        """
        metric: a scalar that goes LOW at the bottom and HIGH at the top (or vice versa).
        We assume:
          - bottom detected when metric <= low_thresh
          - top detected when metric >= high_thresh
        """
        transition = None
        if self.state == "top":
            if metric <= self.low_thresh:
                self.state = "going_down"
                self.frames_in_state = 1
        elif self.state == "going_down":
            self.frames_in_state += 1
            if metric <= self.low_thresh and self.frames_in_state >= self.min_frames:
                self.state = "bottom"
                self.frames_in_state = 0
        elif self.state == "bottom":
            if metric >= self.high_thresh:
                self.state = "going_up"
                self.frames_in_state = 1
        elif self.state == "going_up":
            self.frames_in_state += 1
            if metric >= self.high_thresh and self.frames_in_state >= self.min_frames:
                self.state = "top"
                self.frames_in_state = 0
                self.reps += 1
                transition = "rep"
        return transition

    def reset(self):
        self.state = "top"
        self.frames_in_state = 0
        self.reps = 0

# Initialize counters tuned per exercise
counters = {
    "pushup": RepCounter(low_thresh=75,  high_thresh=150, min_frames=2),  # uses elbow angle
    "squat":  RepCounter(low_thresh=70,  high_thresh=160, min_frames=2),  # uses knee angle
    "pullup": RepCounter(low_thresh=0.40, high_thresh=0.55, min_frames=2),# uses relative wrist height metric
    "plank":  RepCounter(low_thresh=0,   high_thresh=0,   min_frames=9999) # no reps; time/quality only
}

# Smooth feedback queue
feedback_buffer = deque(maxlen=15)

def add_feedback(msg):
    feedback_buffer.append(msg)

def get_feedback():
    if not feedback_buffer:
        return "Ready."
    # Return the most frequent recent message for stability
    vals, counts = np.unique(list(feedback_buffer), return_counts=True)
    return str(vals[np.argmax(counts)])

# -----------------------------
# Exercise-specific metrics
# -----------------------------
def analyze_pushup(landmarks, w, h):
    """
    Metric: left elbow angle (shoulder-elbow-wrist).
    Feedback based on angle thresholds.
    """
    L = mp_pose.PoseLandmark
    sh = get_point(landmarks, L.LEFT_SHOULDER, w, h)
    el = get_point(landmarks, L.LEFT_ELBOW,    w, h)
    wr = get_point(landmarks, L.LEFT_WRIST,    w, h)

    if not visible(sh[2], el[2], wr[2]):
        add_feedback("Move into frame / show left arm.")
        return None, "Poor visibility"

    elbow_ang = angle_3pt(sh, el, wr)  # ~ 160-180 top; ~50-80 bottom
    # Update rep state
    transition = counters["pushup"].update_value(elbow_ang)

    # Feedback
    if elbow_ang > 160:
        add_feedback("Lower your body more.")
    elif elbow_ang < 70:
        add_feedback("Push back up.")
    else:
        add_feedback("Good control.")

    if transition == "rep":
        add_feedback("Good rep!")

    return elbow_ang, None

def analyze_squat(landmarks, w, h):
    """
    Metric: left knee angle (hip-knee-ankle).
    """
    L = mp_pose.PoseLandmark
    hip = get_point(landmarks, L.LEFT_HIP,   w, h)
    knee = get_point(landmarks, L.LEFT_KNEE, w, h)
    ank = get_point(landmarks, L.LEFT_ANKLE, w, h)

    if not visible(hip[2], knee[2], ank[2]):
        add_feedback("Show full lower body.")
        return None, "Poor visibility"

    knee_ang = angle_3pt(hip, knee, ank)  # ~ >160 top; ~60-90 bottom
    transition = counters["squat"].update_value(knee_ang)

    if knee_ang > 160:
        add_feedback("Go lower.")
    elif knee_ang < 70:
        add_feedback("Drive up through heels.")
    else:
        add_feedback("Nice depth.")

    if transition == "rep":
        add_feedback("Great squat rep!")

    return knee_ang, None

def analyze_pullup(landmarks, w, h):
    """
    Metric: normalized wrist height vs shoulder (ratio smaller => higher pull).
    We'll use (wrist_y - shoulder_y), normalized by torso length (shoulder-hip).
    Lower value => you pulled up. We'll invert to fit counter's thresholds.
    For counter:
      - low_thresh ~ 0.40 (at top)
      - high_thresh ~ 0.55 (at bottom)
    """
    L = mp_pose.PoseLandmark
    sh = get_point(landmarks, L.LEFT_SHOULDER, w, h)
    hp = get_point(landmarks, L.LEFT_HIP,      w, h)
    wr = get_point(landmarks, L.LEFT_WRIST,    w, h)

    if not visible(sh[2], hp[2], wr[2]):
        add_feedback("Show upper body and bar area.")
        return None, "Poor visibility"

    torso = max(abs(hp[1] - sh[1]), 1e-6)
    rel = (wr[1] - sh[1]) / torso  # smaller => higher pull
    rel = float(np.clip(rel, -1.0, 2.0))
    # For our rep counter, smaller == bottom; but we defined low as bottom.
    # This matches: at top (chin-up), rel ~ 0.3-0.4 (<= low_thresh).
    transition = counters["pullup"].update_value(rel)

    if rel <= 0.40:
        add_feedback("Hold top—great pull!")
    elif rel >= 0.55:
        add_feedback("Pull up!")
    else:
        add_feedback("Controlled range.")

    if transition == "rep":
        add_feedback("Strong pull-up rep!")

    return rel, None

def analyze_plank(landmarks, w, h):
    """
    Metric: body straightness via angle shoulder-hip-ankle (~180 straight).
    No reps; provide posture feedback.
    """
    L = mp_pose.PoseLandmark
    sh = get_point(landmarks, L.LEFT_SHOULDER, w, h)
    hp = get_point(landmarks, L.LEFT_HIP,      w, h)
    ank = get_point(landmarks, L.LEFT_ANKLE,   w, h)

    if not visible(sh[2], hp[2], ank[2]):
        add_feedback("Show full side profile.")
        return None, "Poor visibility"

    body_ang = angle_3pt(sh, hp, ank)
    if body_ang < 165:
        add_feedback("Keep hips up—straighten body.")
    else:
        add_feedback("Solid plank position.")

    return body_ang, None

ANALYZERS = {
    "pushup": analyze_pushup,
    "squat":  analyze_squat,
    "pullup": analyze_pullup,
    "plank":  analyze_plank
}

# -----------------------------
# Main loop
# -----------------------------
def main():
    global exercise_idx
    print("Form Tracker (no frontend)")
    print("Keys: 1=Push-up, 2=Pull-up, 3=Squat, 4=Plank, r=Reset, q=Quit")

    # Some Windows setups need CAP_DSHOW
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access webcam.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            exercise = EXERCISES[exercise_idx]
            reps = counters[exercise].reps

            if res.pose_landmarks:
                # draw skeleton
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(thickness=2)
                )
                # analyze
                try:
                    metric, err = ANALYZERS[exercise](res.pose_landmarks.landmark, w, h)
                except Exception as e:
                    metric, err = None, f"Analysis error: {e}"
                    add_feedback("Adjust position / try again.")

                # show metric (optional)
                if metric is not None and exercise in ["pushup", "squat"]:
                    cv2.putText(frame, f"Angle: {metric:.1f} deg", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
                elif metric is not None and exercise == "pullup":
                    cv2.putText(frame, f"Rel height: {metric:.2f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
                elif metric is not None and exercise == "plank":
                    cv2.putText(frame, f"Body angle: {metric:.1f} deg", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
                if err:
                    cv2.putText(frame, err, (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
            else:
                add_feedback("Searching for body…")

            # headers & feedback
            cv2.putText(frame, f"Exercise: {exercise.upper()}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
            cv2.putText(frame, f"Reps: {reps}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)
            cv2.putText(frame, get_feedback(),
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 0), 2)

            cv2.imshow("Form Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                exercise_idx = int(chr(key)) - 1
                add_feedback(f"Switched to {EXERCISES[exercise_idx].upper()}")
            elif key == ord('r'):
                for c in counters.values():
                    c.reset()
                add_feedback("Counters reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
