import cv2
import numpy as np
import mediapipe as mp
import time, os, json, math, csv
from collections import deque, defaultdict
from datetime import datetime

# ---------- Optional ML ----------
USE_ML = True
try:
    import joblib
    EX_MODEL = joblib.load("exercise_model.pkl")       # (optional) auto exercise
    FORM_MODEL = None
    if os.path.exists("form_model.pkl"):
        FORM_MODEL = joblib.load("form_model.pkl")     # (optional) per-joint correctness
except Exception:
    USE_ML = False
    EX_MODEL, FORM_MODEL = None, None

# ---------- Pose Setup ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark

# ---------- Config & Thresholds (quickly adjustable at runtime) ----------
CFG = {
    "pushup": {
        "nice_elbow_min": 80,    # in-range lower bound (degrees)
        "nice_elbow_max": 150,   # in-range upper bound
        "min_depth_ratio": 0.35  # shoulder-to-wrist / arm-length
    },
    "squat": {
        "nice_knee_min": 80,
        "nice_knee_max": 140,
        "stance_min_hip_ratio": 0.8,  # stance width >= 0.8*hip width
        "stance_max_hip_ratio": 2.0,
        "foot_angle_max_deg": 40
    },
    "pullup": {
        "top_rel": 0.40,   # <= top
        "bottom_rel": 0.60 # >= bottom
    },
    "plank": {
        "body_straight_min_deg": 165
    }
}

# quick hard/soft toggle for “nice rep”:
ANGLE_TWEAK_STEP = 5  # degrees per key press

# ---------- Utilities ----------
def to_np(pt):
    return np.array(pt[:2], dtype=np.float32)

def angle_3pt(a, b, c):
    a = to_np(a); b = to_np(b); c = to_np(c)
    ab = a - b; cb = c - b
    denom = (np.linalg.norm(ab)*np.linalg.norm(cb) + 1e-6)
    cosang = np.clip(np.dot(ab, cb)/denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    if ang > 180: ang = 360 - ang
    return float(ang)

def dist(a, b):
    return float(np.linalg.norm(to_np(a)-to_np(b)))

def get(lms, idx, w, h):
    lm = lms[idx.value]
    return (lm.x*w, lm.y*h, lm.visibility)

def ok_vis(*vals, t=0.5):
    return all(v is not None and v >= t for v in vals)

def unit_vec(a, b):
    v = to_np(b) - to_np(a); n = np.linalg.norm(v) + 1e-6
    return (v / n).tolist()

def cosine_sim(u, v):
    u = np.array(u); v = np.array(v)
    denom = (np.linalg.norm(u)*np.linalg.norm(v)+1e-9)
    return float(np.dot(u, v)/denom)

# ---------- Session / Set Tracking ----------
class RepCounter:
    def __init__(self, low, high, min_frames=2):
        self.low = low
        self.high = high
        self.min_frames = min_frames
        self.state = "top"
        self.frames = 0
        self.reps = 0

    def update(self, metric):
        tr = None
        if self.state == "top":
            if metric <= self.low:
                self.state = "going_down"; self.frames = 1
        elif self.state == "going_down":
            self.frames += 1
            if metric <= self.low and self.frames >= self.min_frames:
                self.state = "bottom"; self.frames = 0
        elif self.state == "bottom":
            if metric >= self.high:
                self.state = "going_up"; self.frames = 1
        elif self.state == "going_up":
            self.frames += 1
            if metric >= self.high and self.frames >= self.min_frames:
                self.state = "top"; self.frames = 0
                self.reps += 1; tr = "rep"
        return tr

    def reset(self):
        self.state = "top"; self.frames = 0; self.reps = 0

# global counters (heuristic defaults)
COUNTERS = {
    "pushup": RepCounter(low=75, high=150, min_frames=2),   # elbow angle
    "squat":  RepCounter(low=70, high=160, min_frames=2),   # knee angle
    "pullup": RepCounter(low=0.40, high=0.58, min_frames=2) # rel wrist height
}

# per-set logging & features
class WorkoutLogger:
    def __init__(self):
        self.reset_all()
        os.makedirs("sets", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def reset_all(self):
        self.workout = []  # list of reps (each: dict)
        self.active_set = False
        self.set_frames = 0
        self.set_name = None
        self.rep_buffer = []  # collect per-frame features during a single rep
        self.best_rep_vector = None

    def start_set(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.set_name = f"set_{ts}"
        self.active_set = True
        self.set_frames = 0
        self.rep_buffer.clear()
        # video recorder created lazily when first frame arrives (to know size)
        self.rec = None
        self.improvements = defaultdict(int)

    def end_set(self):
        self.active_set = False
        # close video
        if hasattr(self, "rec") and self.rec is not None:
            self.rec.release()
            self.rec = None
        # save JSON and CSV
        json_path = os.path.join("logs", f"{self.set_name}.json")
        csv_path  = os.path.join("logs", f"{self.set_name}.csv")
        with open(json_path, "w") as f:
            json.dump(self.workout, f, indent=2)
        if self.workout:
            keys = sorted(set().union(*[w.keys() for w in self.workout]))
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in self.workout:
                    w.writerow(row)
        # build improvements list
        improve_list = sorted(self.improvements.items(), key=lambda x: -x[1])
        return json_path, csv_path, improve_list

    def ensure_recorder(self, frame):
        if getattr(self, "rec", None) is None:
            h, w = frame.shape[:2]
            out_path = os.path.join("sets", f"{self.set_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.rec = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
            self.set_video_path = out_path

    def log_frame(self, frame):
        self.set_frames += 1
        self.ensure_recorder(frame)
        self.rec.write(frame)

    def begin_rep(self):
        self.rep_buffer.clear()

    def end_rep(self, meta):
        # aggregate rep features
        if not self.rep_buffer: 
            return
        feat = np.mean(np.array(self.rep_buffer), axis=0).tolist()
        meta["rep_feature"] = feat
        # similarity vs best
        if self.best_rep_vector is None:
            self.best_rep_vector = feat
            meta["rep_similarity"] = 1.0
        else:
            meta["rep_similarity"] = cosine_sim(feat, self.best_rep_vector)
            # keep the best example as the one with highest mean score (heuristic)
            if meta.get("form_score", 0) > 0.9 and meta["rep_similarity"] > 0.95:
                self.best_rep_vector = feat
        self.workout.append(meta)

    def add_frame_feature(self, vec):
        # vec: list of numeric features this frame
        self.rep_buffer.append(vec)

    def nudge_issue(self, key):
        self.improvements[key] += 1

LOGGER = WorkoutLogger()

# ---------- Anthropometrics (auto-calibration at set start) ----------
class Anthro:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.calibrated = False
        self.stats = {}

    def maybe_collect(self, lms, w, h, max_samples=30):
        if self.calibrated:
            return
        # collect femur (hip->knee), tibia (knee->ankle), hip width (L-R hip)
        lh = get(lms, L.LEFT_HIP, w, h); rh = get(lms, L.RIGHT_HIP, w, h)
        lk = get(lms, L.LEFT_KNEE, w, h); rk = get(lms, L.RIGHT_KNEE, w, h)
        la = get(lms, L.LEFT_ANKLE, w, h); ra = get(lms, L.RIGHT_ANKLE, w, h)
        if not ok_vis(lh[2], rh[2], lk[2], rk[2], la[2], ra[2], t=0.4):
            return
        femur_L = dist(lh, lk); femur_R = dist(rh, rk)
        tibia_L = dist(lk, la); tibia_R = dist(rk, ra)
        hip_width = dist(lh, rh)
        self.samples.append([femur_L, femur_R, tibia_L, tibia_R, hip_width])
        if len(self.samples) >= max_samples:
            arr = np.array(self.samples)
            m = arr.mean(axis=0)
            self.stats = {
                "femur_L": m[0], "femur_R": m[1],
                "tibia_L": m[2], "tibia_R": m[3],
                "hip_width": m[4]
            }
            self.calibrated = True

ANTHRO = Anthro()

# ---------- Exercise Heuristics (fallback) ----------
def heuristic_exercise(landmarks, w, h):
    # simple rule cues: largest motion over short window
    # Here we check elbow vs knee dynamics to guess
    sh = get(landmarks, L.LEFT_SHOULDER, w, h)
    el = get(landmarks, L.LEFT_ELBOW,    w, h)
    wr = get(landmarks, L.LEFT_WRIST,    w, h)
    hp = get(landmarks, L.LEFT_HIP,      w, h)
    kn = get(landmarks, L.LEFT_KNEE,     w, h)
    an = get(landmarks, L.LEFT_ANKLE,    w, h)
    if not ok_vis(sh[2], el[2], wr[2], hp[2], kn[2], an[2], t=0.3):
        return "plank"

    elbow_ang = angle_3pt(sh, el, wr)
    knee_ang = angle_3pt(hp, kn, an)

    # crude:
    if knee_ang < 140:      # lots of knee flexion
        return "squat"
    if elbow_ang < 120:     # elbow flexing repeatedly
        return "pushup"
    # wrist vs shoulder height (pullup)
    torso = max(abs(hp[1]-sh[1]), 1e-6)
    rel = (wr[1]-sh[1])/torso
    if rel < 0.5: return "pullup"
    return "plank"

# ---------- Analysis per Exercise ----------
def analyze_pushup(lms, w, h, overlay, notes):
    ls = get(lms, L.LEFT_SHOULDER, w, h)
    le = get(lms, L.LEFT_ELBOW,    w, h)
    lw = get(lms, L.LEFT_WRIST,    w, h)
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    re = get(lms, L.RIGHT_ELBOW,    w, h)
    rw = get(lms, L.RIGHT_WRIST,    w, h)

    # both-hand vectors (elbow->wrist)
    vL = unit_vec(le, lw); vR = unit_vec(re, rw)

    # elbow angles (avg both arms if visible)
    okL = ok_vis(ls[2], le[2], lw[2]); okR = ok_vis(rs[2], re[2], rw[2])
    elbow_L = angle_3pt(ls, le, lw) if okL else None
    elbow_R = angle_3pt(rs, re, rw) if okR else None
    elbow_mean = None
    if elbow_L and elbow_R: elbow_mean = (elbow_L + elbow_R)/2.0
    elif elbow_L: elbow_mean = elbow_L
    elif elbow_R: elbow_mean = elbow_R

    # push-up depth ratio: vertical (shoulder_y - wrist_y) / arm_length
    def arm_len(s, e, w_):
        return (dist(s,e)+dist(e,w_))
    depth_ratio = None
    if elbow_mean is not None:
        # use better-visible side
        if okL:
            armL = arm_len(ls, le, lw)
            depth_ratio = max(0.0, (ls[1]-lw[1]))/(armL+1e-6)
        elif okR:
            armR = arm_len(rs, re, rw)
            depth_ratio = max(0.0, (rs[1]-rw[1]))/(armR+1e-6)

    # suggestion scoring
    form_score = 1.0
    if elbow_mean is not None:
        if not (CFG["pushup"]["nice_elbow_min"] <= elbow_mean <= CFG["pushup"]["nice_elbow_max"]):
            notes.append("Elbow angle out of optimal range")
            form_score -= 0.2
        if depth_ratio is not None and depth_ratio < CFG["pushup"]["min_depth_ratio"]:
            notes.append("Increase push-up depth")
            form_score -= 0.2

    # hands symmetry check
    hands_sym = cosine_sim(vL, vR) if (okL and okR) else None
    if hands_sym is not None and hands_sym < 0.8:
        notes.append("Hand/forearm symmetry off")
        form_score -= 0.1

    # frame-level feature vector (for similarity)
    feat = [
        elbow_mean if elbow_mean is not None else 0.0,
        depth_ratio if depth_ratio is not None else 0.0,
        hands_sym   if hands_sym is not None   else 1.0
    ]
    # overlay text
    if elbow_mean is not None:
        cv2.putText(overlay, f"Elbow: {elbow_mean:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    if depth_ratio is not None:
        cv2.putText(overlay, f"Depth: {depth_ratio:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    return feat, form_score

def analyze_squat(lms, w, h, overlay, notes):
    lh = get(lms, L.LEFT_HIP, w, h); rh = get(lms, L.RIGHT_HIP, w, h)
    lk = get(lms, L.LEFT_KNEE, w, h); rk = get(lms, L.RIGHT_KNEE, w, h)
    la = get(lms, L.LEFT_ANKLE, w, h); ra = get(lms, L.RIGHT_ANKLE, w, h)
    lheel = get(lms, L.LEFT_HEEL, w, h); rheel = get(lms, L.RIGHT_HEEL, w, h)
    ltoe  = get(lms, L.LEFT_FOOT_INDEX, w, h); rtoe  = get(lms, L.RIGHT_FOOT_INDEX, w, h)

    # femur & tibia lengths
    femur_L = dist(lh, lk); femur_R = dist(rh, rk)
    tibia_L = dist(lk, la); tibia_R = dist(rk, ra)

    # knee angles (avg)
    knee_L = angle_3pt(lh, lk, la) if ok_vis(lh[2], lk[2], la[2]) else None
    knee_R = angle_3pt(rh, rk, ra) if ok_vis(rh[2], rk[2], ra[2]) else None
    knee_mean = None
    if knee_L and knee_R: knee_mean = (knee_L+knee_R)/2
    elif knee_L: knee_mean = knee_L
    elif knee_R: knee_mean = knee_R

    # foot position: stance width normalized by hip width; foot angle
    hip_w = dist(lh, rh) if ok_vis(lh[2], rh[2]) else None
    stance_w = dist(la, ra) if ok_vis(la[2], ra[2]) else None
    stance_ratio = (stance_w / (hip_w+1e-6)) if (hip_w and stance_w) else None

    def foot_angle(heel, toe):
        v = to_np(toe) - to_np(heel)
        ang = np.degrees(np.arctan2(v[1], v[0]))
        # angle relative to x-axis; we just care about absolute flare
        return abs(ang)
    foot_ang_L = foot_angle(lheel, ltoe) if ok_vis(lheel[2], ltoe[2], t=0.3) else None
    foot_ang_R = foot_angle(rheel, rtoe) if ok_vis(rheel[2], rtoe[2], t=0.3) else None
    foot_ang_mean = None
    if foot_ang_L and foot_ang_R: foot_ang_mean = (foot_ang_L+foot_ang_R)/2
    elif foot_ang_L: foot_ang_mean = foot_ang_L
    elif foot_ang_R: foot_ang_mean = foot_ang_R

    # form scoring
    form_score = 1.0
    if knee_mean is not None and not (CFG["squat"]["nice_knee_min"] <= knee_mean <= CFG["squat"]["nice_knee_max"]):
        notes.append("Adjust squat depth (knee angle)")
        form_score -= 0.2
    if stance_ratio is not None:
        if stance_ratio < CFG["squat"]["stance_min_hip_ratio"]:
            notes.append("Widen stance slightly")
            form_score -= 0.1
        if stance_ratio > CFG["squat"]["stance_max_hip_ratio"]:
            notes.append("Narrow stance slightly")
            form_score -= 0.1
    if foot_ang_mean is not None and foot_ang_mean > CFG["squat"]["foot_angle_max_deg"]:
        notes.append("Reduce foot flare")
        form_score -= 0.05

    # features
    feat = [
        knee_mean if knee_mean is not None else 0.0,
        stance_ratio if stance_ratio is not None else 1.0,
        foot_ang_mean if foot_ang_mean is not None else 0.0,
        femur_L, femur_R, tibia_L, tibia_R
    ]

    # overlay
    if knee_mean is not None:
        cv2.putText(overlay, f"Knee: {knee_mean:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    if stance_ratio is not None:
        cv2.putText(overlay, f"Stance/Hip: {stance_ratio:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    if foot_ang_mean is not None:
        cv2.putText(overlay, f"Foot angle: {foot_ang_mean:.0f} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)

    return feat, form_score

def analyze_pullup(lms, w, h, overlay, notes):
    sh = get(lms, L.LEFT_SHOULDER, w, h)
    hp = get(lms, L.LEFT_HIP,      w, h)
    wr = get(lms, L.LEFT_WRIST,    w, h)
    torso = max(abs(hp[1]-sh[1]), 1e-6)
    rel = (wr[1]-sh[1])/torso  # smaller => higher pull
    form_score = 1.0
    if rel > CFG["pullup"]["bottom_rel"]:
        notes.append("Pull higher (reach the bar)")
        form_score -= 0.2
    # features
    feat = [rel]
    cv2.putText(overlay, f"Rel height: {rel:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    return feat, form_score

def analyze_plank(lms, w, h, overlay, notes):
    sh = get(lms, L.LEFT_SHOULDER, w, h)
    hp = get(lms, L.LEFT_HIP,      w, h)
    an = get(lms, L.LEFT_ANKLE,    w, h)
    body_ang = angle_3pt(sh, hp, an)
    form_score = 1.0
    if body_ang < CFG["plank"]["body_straight_min_deg"]:
        notes.append("Raise hips / keep straight line")
        form_score -= 0.2
    feat = [body_ang]
    cv2.putText(overlay, f"Body angle: {body_ang:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,230,255), 2)
    return feat, form_score

ANALYZE = {
    "pushup": analyze_pushup,
    "squat":  analyze_squat,
    "pullup": analyze_pullup,
    "plank":  analyze_plank
}

# ---------- ML Predictors ----------
EX_BUF = deque(maxlen=12)
def predict_exercise_ml(lms):
    row = []
    for lm in lms:
        row += [lm.x, lm.y]
    lab = EX_MODEL.predict([row])[0]
    EX_BUF.append(lab)
    # majority vote for stability
    labs, cnts = np.unique(list(EX_BUF), return_counts=True)
    return labs[np.argmax(cnts)]

def score_form_ml(lms):
    if FORM_MODEL is None:
        return None, {}
    row = []
    for lm in lms:
        row += [lm.x, lm.y]
    proba = FORM_MODEL.predict_proba([row])[0]
    score = float(max(proba))
    # optional: class-specific hints (if model supports it)
    return score, {}

# ---------- Main ----------
def main():
    print("Keys: s=start set, e=end set & save, m=toggle ML auto-ex, [ / ] tweak angles, q=quit")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not found"); return

    auto_ex = USE_ML and (EX_MODEL is not None)
    current_ex = "plank"  # default until predicted
    pending_switch = deque(maxlen=8)
    in_rep = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            view = frame.copy()
            h, w = frame.shape[:2]

            # Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            notes = []
            feat_vec = [0.0]  # default
            form_score = 1.0

            if res.pose_landmarks:
                mp_draw.draw_landmarks(view, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_draw.DrawingSpec(thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(thickness=2))

                # Anthro calibration at start of set
                if LOGGER.active_set and not ANTHRO.calibrated:
                    ANTHRO.maybe_collect(res.pose_landmarks, w, h)

                # Exercise selection
                if auto_ex:
                    lab = predict_exercise_ml(res.pose_landmarks.landmark)
                else:
                    # heuristic fallback
                    lab = heuristic_exercise(res.pose_landmarks.landmark, w, h)
                # Smooth the switch
                if lab != current_ex:
                    pending_switch.append(lab)
                    if len(pending_switch) == pending_switch.maxlen and len(set(pending_switch)) == 1:
                        current_ex = lab
                        pending_switch.clear()
                else:
                    pending_switch.clear()

                # Analyze per exercise
                feat_vec, form_score_local = ANALYZE[current_ex](res.pose_landmarks.landmark, w, h, view, notes)
                form_score = min(form_score, form_score_local)

                # Optional ML scoring for form
                if FORM_MODEL is not None:
                    ml_score, part_hints = score_form_ml(res.pose_landmarks.landmark)
                    if ml_score is not None:
                        form_score = 0.5*form_score + 0.5*ml_score

                # Rep logic
                if current_ex == "pushup":
                    # metric: elbow angle mean (we computed inside analyze; reuse feat[0])
                    metric = feat_vec[0]
                    if metric > 0:
                        tr = COUNTERS["pushup"].update(metric)
                        if tr == "rep":
                            LOGGER.end_rep({
                                "exercise": current_ex,
                                "time": time.time(),
                                "form_score": form_score
                            })
                            in_rep = False
                        else:
                            if not in_rep and COUNTERS["pushup"].state == "bottom":
                                LOGGER.begin_rep(); in_rep = True
                elif current_ex == "squat":
                    metric = feat_vec[0]  # knee angle
                    if metric > 0:
                        tr = COUNTERS["squat"].update(metric)
                        if tr == "rep":
                            LOGGER.end_rep({
                                "exercise": current_ex,
                                "time": time.time(),
                                "form_score": form_score
                            })
                            in_rep = False
                        else:
                            if not in_rep and COUNTERS["squat"].state == "bottom":
                                LOGGER.begin_rep(); in_rep = True
                elif current_ex == "pullup":
                    metric = feat_vec[0]  # rel height
                    tr = COUNTERS["pullup"].update(metric)
                    if tr == "rep":
                        LOGGER.end_rep({
                            "exercise": current_ex,
                            "time": time.time(),
                            "form_score": form_score
                        })
                        in_rep = False
                    else:
                        if not in_rep and COUNTERS["pullup"].state == "going_down":
                            LOGGER.begin_rep(); in_rep = True
                elif current_ex == "plank":
                    # no reps, still collect per-frame features as “segments”
                    pass

                # Per-frame features into rep buffer (for similarity later)
                if LOGGER.active_set and in_rep:
                    # enrich with a few stable extras
                    extras = [form_score]
                    if ANTHRO.calibrated:
                        extras += [ANTHRO.stats["femur_L"], ANTHRO.stats["tibia_L"], ANTHRO.stats["hip_width"]]
                    LOGGER.add_frame_feature(feat_vec + extras)

                # Nudge improvement counters from notes
                if LOGGER.active_set:
                    for n in notes:
                        LOGGER.nudge_issue(n)

            # HUD
            cv2.putText(view, f"Exercise: {current_ex.upper()}  ML:{'ON' if auto_ex else 'OFF'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,0), 2)
            reps_str = " | ".join([f"{k}:{COUNTERS[k].reps}" for k in COUNTERS])
            cv2.putText(view, f"Reps [{reps_str}]", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,220,220), 2)
            if LOGGER.active_set:
                cv2.putText(view, f"REC ●  {LOGGER.set_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                LOGGER.log_frame(view)

            # Tips
            y0 = 240
            if ANTHRO.calibrated:
                cv2.putText(view, f"HipW:{ANTHRO.stats['hip_width']:.0f}  FemurL:{ANTHRO.stats['femur_L']:.0f}  TibiaL:{ANTHRO.stats['tibia_L']:.0f}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,200,255), 2)
            # Show a few last notes
            if notes:
                for i, n in enumerate(notes[:3]):
                    cv2.putText(view, f"• {n}", (10, y0+30+24*i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,220,255), 2)

            cv2.imshow("Form Tracker Pro", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # start set
                if not LOGGER.active_set:
                    LOGGER.start_set()
                    ANTHRO.reset()
                    for c in COUNTERS.values(): c.reset()
                    in_rep = False
            elif key == ord('e'):  # end set
                if LOGGER.active_set:
                    jpath, cpath, improvements = LOGGER.end_set()
                    LOGGER.reset_all()
                    # show summary in console
                    print("\n=== Set Saved ===")
                    print("Video:", getattr(LOGGER, "set_video_path", "(n/a)"))
                    print("JSON :", jpath)
                    print("CSV  :", cpath)
                    print("Top improvements:")
                    for k, v in improvements[:10]:
                        print(f"- {k}  (x{v})")
                    print("=================\n")
            elif key == ord('m'):
                auto_ex = not auto_ex and (EX_MODEL is not None)
            elif key == ord('['):  # loosen “nice rep”
                CFG["pushup"]["nice_elbow_min"] = max(10, CFG["pushup"]["nice_elbow_min"]-ANGLE_TWEAK_STEP)
                CFG["pushup"]["nice_elbow_max"] = min(175, CFG["pushup"]["nice_elbow_max"]+ANGLE_TWEAK_STEP)
                CFG["squat"]["nice_knee_min"]   = max(40, CFG["squat"]["nice_knee_min"]-ANGLE_TWEAK_STEP)
                CFG["squat"]["nice_knee_max"]   = min(175, CFG["squat"]["nice_knee_max"]+ANGLE_TWEAK_STEP)
            elif key == ord(']'):  # tighten “nice rep”
                CFG["pushup"]["nice_elbow_min"] = min(CFG["pushup"]["nice_elbow_max"]-10, CFG["pushup"]["nice_elbow_min"]+ANGLE_TWEAK_STEP)
                CFG["pushup"]["nice_elbow_max"] = max(CFG["pushup"]["nice_elbow_min"]+10, CFG["pushup"]["nice_elbow_max"]-ANGLE_TWEAK_STEP)
                CFG["squat"]["nice_knee_min"]   = min(CFG["squat"]["nice_knee_max"]-10, CFG["squat"]["nice_knee_min"]+ANGLE_TWEAK_STEP)
                CFG["squat"]["nice_knee_max"]   = max(CFG["squat"]["nice_knee_min"]+10, CFG["squat"]["nice_knee_max"]-ANGLE_TWEAK_STEP)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
