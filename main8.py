import cv2
import numpy as np
import mediapipe as mp
import time, warnings, threading
from collections import deque
import pyttsx3

# ---------- TEXT TO SPEECH ----------
engine = pyttsx3.init()
engine.setProperty('rate', 185)
engine.setProperty('volume', 1.0)

def speak_async(text):
    """Speak text asynchronously (non-blocking)."""
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ---------- Optional ML ----------
USE_ML = True
try:
    import joblib
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    EX_MODEL = joblib.load("exercise_model.pkl")
except Exception as e:
    print("⚠️ ML model load failed:", e)
    USE_ML = False
    EX_MODEL = None

# ---------- Pose Setup ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark

# ---------- Config ----------
CFG = {
    "pushup": {"min_down": 120, "max_up": 160},
    # Squat thresholds here are for *depth phase gating* (hip depth). The stance judgment happens at the bottom.
    "squat":  {"min_down": 60, "max_up": 130},   # <-- from your working squat code
}

# ---------- Utilities ----------
def to_np(pt): return np.array(pt[:2], dtype=np.float32)

def angle_3pt(a,b,c):
    a,b,c = to_np(a),to_np(b),to_np(c)
    ab,cb = a-b,c-b
    denom = (np.linalg.norm(ab)*np.linalg.norm(cb)+1e-6)
    cosang = np.clip(np.dot(ab,cb)/denom,-1.0,1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(ang if ang<=180 else 360-ang)

def get(lms,idx,w,h):
    lm = lms[idx.value]
    # return (x, y, visibility)
    return (lm.x*w, lm.y*h, lm.visibility)

def ok_vis(*vals,t=0.5):
    return all(v is not None and v>=t for v in vals)

# ---------- Distance helpers (kept from your second file; used for tips) ----------
def stance_width(lms, w, h):
    la = get(lms, L.LEFT_ANKLE, w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    if not ok_vis(la[2], ra[2]): return None
    return abs(la[0] - ra[0])

def knee_distance(lms, w, h):
    lk = get(lms, L.LEFT_KNEE, w, h)
    rk = get(lms, L.RIGHT_KNEE, w, h)
    if not ok_vis(lk[2], rk[2]): return None
    return abs(lk[0] - rk[0])

def estimate_camera_distance(lms, w, h):
    try:
        sh = get(lms, L.LEFT_SHOULDER, w, h)
        an = get(lms, L.LEFT_ANKLE, w, h)
        if not ok_vis(sh[2], an[2]):
            return 1.0
        body_height = abs(sh[1] - an[1])
        norm_height = max(body_height / h, 1e-3)
        distance_m = np.clip(1.5 / (norm_height * 4.0), 0.75, 2.5)
        return float(distance_m)
    except:
        return 1.0

# ---------- Rep Counter (generic; used for pushups only) ----------
class RepCounter:
    def __init__(self, low, high, min_frames=2):
        self.low, self.high, self.min_frames = low, high, min_frames
        self.state, self.frames, self.reps = "top", 0, 0
        self.bad_reps, self.last_rep_time = 0, 0
        self.last_feedback = ""

    def update(self, metric):
        if metric is None: return None
        now = time.time()
        tr = None
        if self.state == "top":
            if metric <= self.low:
                self.state, self.frames = "going_down", 1
        elif self.state == "going_down":
            self.frames += 1
            if metric <= self.low and self.frames >= self.min_frames:
                self.state, self.frames = "bottom", 0
        elif self.state == "bottom":
            if metric >= self.high:
                self.state, self.frames = "going_up", 1
        elif self.state == "going_up":
            self.frames += 1
            if metric >= self.high and self.frames >= self.min_frames:
                if now - self.last_rep_time > 0.6:
                    self.state, self.frames = "top", 0
                    self.reps += 1
                    self.last_rep_time = now
                    tr = "rep"
        return tr

    def reset(self):
        self.state, self.frames, self.reps, self.bad_reps = "top", 0, 0, 0
        self.last_feedback = ""

# ---------- Counters ----------
COUNTERS = {
    "pushup": RepCounter(low=70, high=155),   # keep your pushup detection as-is
}
# Note: squat will use its *own* specialized counter below.

# ---------- Smoothing ----------
SMOOTH = {k: deque(maxlen=5) for k in COUNTERS}
def smooth_metric(name,val):
    if val is None: return None
    q = SMOOTH[name]; q.append(float(val))
    return float(np.mean(q))

# ---------- ANALYSIS: PUSHUP (unchanged; keep what works for you) ----------
def analyze_pushup(lms,w,h):
    ls,le,lw = get(lms,L.LEFT_SHOULDER,w,h),get(lms,L.LEFT_ELBOW,w,h),get(lms,L.LEFT_WRIST,w,h)
    rs,re,rw = get(lms,L.RIGHT_SHOULDER,w,h),get(lms,L.RIGHT_ELBOW,w,h),get(lms,L.RIGHT_WRIST,w,h)
    elbows=[]
    if ok_vis(ls[2],le[2],lw[2]): elbows.append(angle_3pt(ls,le,lw))
    if ok_vis(rs[2],re[2],rw[2]): elbows.append(angle_3pt(rs,re,rw))
    if not elbows: return None
    return float(np.mean(elbows))

# ---------- ANALYSIS: SQUAT (ported from your working first script) ----------
def analyze_squat_depth_and_ratio(lms, w, h):
    """Return: depth, ratio, hip_y, knee_y.
       depth = (avg_ankle_y - avg_hip_y): larger => deeper.
       ratio = knee_distance / shoulder_distance (live).
    """
    rh = get(lms, L.RIGHT_HIP, w, h)
    lh = get(lms, L.LEFT_HIP,  w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    la = get(lms, L.LEFT_ANKLE,  w, h)
    rk = get(lms, L.RIGHT_KNEE, w, h)
    lk = get(lms, L.LEFT_KNEE,  w, h)
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    ls = get(lms, L.LEFT_SHOULDER,  w, h)

    hip_y = knee_y = depth = ratio = None

    if ok_vis(rh[2], lh[2], ra[2], la[2]):
        hip_y   = (rh[1] + lh[1]) / 2.0
        ankle_y = (ra[1] + la[1]) / 2.0
        depth   = ankle_y - hip_y

    if ok_vis(rk[2], lk[2]):
        knee_y  = (rk[1] + lk[1]) / 2.0

    if ok_vis(ls[2], rs[2], lk[2], rk[2]):
        shoulder_dist = np.linalg.norm(np.array(ls[:2]) - np.array(rs[:2]))
        knee_dist     = np.linalg.norm(np.array(lk[:2]) - np.array(rk[:2]))
        ratio = (knee_dist / shoulder_dist) if shoulder_dist != 0 else None

    return depth, ratio, hip_y, knee_y

def stance_feedback(ratio):
    if ratio is None:
        return ""
    if ratio < 0.83:
        return "Bring your knees in"
    elif ratio > 1.40:
        return "Spread your knees more"
    else:
        return "Perfect! Keep going!"

# ---------- Specialized SquatCounter (from your working logic) ----------
class SquatCounter:
    """Counts ONLY 'perfect' reps:
       - Capture stance *at bottom* (pelvis below knees).
       - On rise above knees, count rep only if bottom stance was Perfect.
    """
    def __init__(self, min_down=CFG["squat"]["min_down"], max_up=CFG["squat"]["max_up"], min_frames=2):
        self.low, self.high, self.min_frames = min_down, max_up, min_frames
        self.state = "top"
        self.frames = 0

        self.reps = 0           # good reps only
        self.bad_reps = 0       # bad attempts

        self.bottom_reached = False
        self.max_hip_y = None
        self.bottom_ratio = None
        self.bottom_feedback = ""

        self.last_rep_feedback = ""
        self.last_feedback_time = 0
        self.last_rep_time = 0

    def reset(self):
        self.__init__(self.low, self.high, self.min_frames)

    def update(self, depth, hip_y, knee_y, ratio):
        if depth is None or hip_y is None or knee_y is None:
            return None

        now = time.time()
        below_knee = (hip_y > knee_y)  # image y goes downward

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
                    if self.bottom_feedback and "Perfect" in self.bottom_feedback:
                        self.reps += 1
                        self.last_rep_feedback = "✅ Good squat"
                        speak_async("Good rep")
                    else:
                        self.bad_reps += 1
                        fb = self.bottom_feedback if self.bottom_feedback else "Adjust your stance"
                        self.last_rep_feedback = f"❌ {fb}"
                        speak_async(fb)
                    self.last_feedback_time = now

                self.state = "top"
                self.frames = 0

# instantiate squat counter
SQUAT_COUNTER = SquatCounter()

# ---------- ML prediction (unchanged) ----------
EX_BUF = deque(maxlen=12)
def predict_exercise_ml(lms):
    if not USE_ML or EX_MODEL is None: return "plank"
    row = [val for lm in lms for val in (lm.x,lm.y)]
    lab = EX_MODEL.predict([row])[0]
    lab = str(lab).lower().strip()
    if lab.endswith("s"): lab = lab[:-1]
    mapping = {"pushups":"pushup","squats":"squat","planks":"plank"}
    lab = mapping.get(lab,lab)
    EX_BUF.append(lab)
    labs,cnts = np.unique(list(EX_BUF),return_counts=True)
    return labs[np.argmax(cnts)]

# ---------- Drawing helper ----------
def draw_text(view, text, pos, color=(10,10,10), scale=0.8, thick=2):
    x, y = pos
    cv2.putText(view, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick+3)
    cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ---------- Recommendation (kept for pushup feedback) ----------
def get_recommendation(ex_type, metric, lms, w, h):
    distance = estimate_camera_distance(lms, w, h)
    width = stance_width(lms, w, h)
    knees = knee_distance(lms, w, h)

    center_knee = 65 * (distance / 0.9)
    ideal_knee_min = center_knee - 12.5
    ideal_knee_max = center_knee + 12.5

    if ex_type == "pushup":
        if metric is None: return ""
        if metric > CFG["pushup"]["max_up"]:
            return "Go all the way down"
        elif metric < CFG["pushup"]["min_down"]:
            return "Go all the way up"
        elif 85 < metric < 120:
            return "Adjust arm range — spread out arms"
    elif ex_type == "squat":
        # We now judge squats via the bottom-stance logic, not here.
        return ""
    return ""

# ---------- Main ----------
def main():
    print("Keys: s=start set, e=end set (stats), q=quit")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found.")
        return

    current_ex = "plank"
    pending_switch = deque(maxlen=8)
    last_feedback = ""
    feedback_timer = 0
    bad_last = False

    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            view = frame.copy()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                mp_draw.draw_landmarks(view, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Auto exercise selection (keep)
                if USE_ML and EX_MODEL is not None:
                    lab = predict_exercise_ml(res.pose_landmarks.landmark)
                else:
                    lab = current_ex
                if lab != current_ex:
                    pending_switch.append(lab)
                    if len(pending_switch) == pending_switch.maxlen and len(set(pending_switch)) == 1:
                        current_ex = lab; pending_switch.clear()

                # Distance info (optional overlay)
                distance = estimate_camera_distance(res.pose_landmarks.landmark, w, h)
                knees_px = knee_distance(res.pose_landmarks.landmark, w, h)
                ideal_knee_min = 60 * (distance / 0.9)
                ideal_knee_max = 70 * (distance / 0.9)
                draw_text(view, f"Distance: {distance:.2f} m", (10, 100), (150,100,255), 0.7, 2)
                if knees_px:
                    draw_text(view, f"KneeDist: {int(knees_px)} px (ideal {int(ideal_knee_min)}–{int(ideal_knee_max)})",
                              (10, 125), (120,50,220), 0.7, 2)

                # ---------- PUSHUP PATH (unchanged counting) ----------
                if current_ex == "pushup":
                    metric_raw = analyze_pushup(res.pose_landmarks.landmark, w, h)
                    metric = smooth_metric("pushup", metric_raw)

                    if metric is not None:
                        tr = COUNTERS["pushup"].update(metric)
                        if tr == "rep":
                            reco = get_recommendation("pushup", metric, res.pose_landmarks.landmark, w, h)
                            if reco:
                                speak_async(reco)
                                last_feedback = f"Tip: {reco}"
                                bad_last = True
                                COUNTERS["pushup"].bad_reps += 1
                            else:
                                last_feedback = f"✅ Good pushup rep!"
                                bad_last = False
                                speak_async("Good rep")
                            feedback_timer = time.time()

                # ---------- SQUAT PATH (replaced with robust logic) ----------
                elif current_ex == "squat":
                    depth, ratio, hip_y, knee_y = analyze_squat_depth_and_ratio(
                        res.pose_landmarks.landmark, w, h
                    )

                    # Always show live ratio (color-coded)
                    if ratio is not None:
                        fb = stance_feedback(ratio)
                        color = (0, 200, 0) if "Perfect" in fb else (0, 0, 255) if "Bring" in fb else (255, 200, 0)
                        draw_text(view, f"Knee/Shoulder Ratio: {ratio:.2f}", (10, 155), color, 0.8, 2)

                    # Update squat counter (only counts perfect reps at bottom)
                    SQUAT_COUNTER.update(depth, hip_y, knee_y, ratio)

                    # Show last squat judgment for ~2s after rep completion
                    if time.time() - SQUAT_COUNTER.last_feedback_time < 2 and SQUAT_COUNTER.last_rep_feedback:
                        good = SQUAT_COUNTER.last_rep_feedback.startswith("✅")
                        draw_text(view, SQUAT_COUNTER.last_rep_feedback, (10, 185),
                                  (0,200,0) if good else (0,0,255), 0.8, 2)

            # HUD
            draw_text(view,f"Exercise: {current_ex.upper()}",(10,30),(10,10,10),0.9,2)
            # Pushup reps (good & bad from COUNTERS)
            pu = COUNTERS["pushup"]
            pu_total = pu.reps + pu.bad_reps
            draw_text(view,f"Pushups: Good={pu.reps} Bad={pu.bad_reps}",(10,65),(30,30,30),0.85,2)
            # Squat reps from specialized counter
            draw_text(view,f"Squats:  Good={SQUAT_COUNTER.reps} Bad={SQUAT_COUNTER.bad_reps}",
                      (10,90),(30,30,30),0.85,2)

            # Recent pushup feedback bubble
            if current_ex == "pushup":
                if time.time()-feedback_timer<2 and last_feedback:
                    color = (0,180,0) if not bad_last else (0,0,255)
                    draw_text(view,last_feedback,(10,185),color,0.8,2)

            cv2.imshow("Form Tracker (Pushups stable + Squats robust)", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Reset both counters
                for q in SMOOTH.values(): q.clear()
                for c in COUNTERS.values(): c.reset()
                SQUAT_COUNTER.reset()
                speak_async("Starting new set")
            elif key == ord('e'):
                speak_async("Workout ended. Showing statistics.")
                print("\n===== Session Statistics =====")
                # Pushups
                pu_total = pu.reps + pu.bad_reps
                pu_acc = (pu.reps / pu_total * 100) if pu_total>0 else 0
                print(f"PUSHUP: Good reps={pu.reps}, Bad reps={pu.bad_reps}, Accuracy={pu_acc:.1f}%")
                # Squats
                sq_total = SQUAT_COUNTER.reps + SQUAT_COUNTER.bad_reps
                sq_acc = (SQUAT_COUNTER.reps / sq_total * 100) if sq_total>0 else 0
                print(f"SQUAT : Good reps={SQUAT_COUNTER.reps}, Bad reps={SQUAT_COUNTER.bad_reps}, Accuracy={sq_acc:.1f}%")
                print("==============================\n")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
