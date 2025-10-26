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
    "pushup": {"nice_angle_min": 30, "nice_angle_max": 90},
    "squat": {"min_down": 60, "max_up": 130}, 
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
    return (lm.x*w, lm.y*h, lm.visibility)
def ok_vis(*vals,t=0.5):
    return all(v is not None and v>=t for v in vals)
# ---------- NEW: Check if whole body is in frame ----------
def is_whole_body_visible(lms, w, h, visibility_threshold=0.5):
    """Check if all key body landmarks are visible for squats."""
    key_landmarks = [
        L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
        L.LEFT_HIP, L.RIGHT_HIP,
        L.LEFT_KNEE, L.RIGHT_KNEE,
        L.LEFT_ANKLE, L.RIGHT_ANKLE
    ]
    
    for landmark in key_landmarks:
        lm = get(lms, landmark, w, h)
        if lm[2] < visibility_threshold:
            return False
    
    # Additional check: make sure ankles are visible in frame
    la = get(lms, L.LEFT_ANKLE, w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    
    # Check if ankles are within frame bounds (not cut off)
    if la[1] >= h * 0.95 or ra[1] >= h * 0.95:
        return False
    
    return True
# ---------- Distance helpers ----------
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
# ---------- NEW: Pushup Counter Based on Elbow Y Position ----------
class PushupCounter:
    def __init__(self, min_frames=2):
        self.min_frames = min_frames
        self.state = "up"  # "up", "going_down", "down", "going_up"
        self.frames = 0
        self.reps = 0
        self.bad_reps = 0
        self.last_rep_time = 0
        self.bottom_form_score = 1.0  # Track form at bottom position
        self.last_rep_feedback = ""
        self.feedback_timer = 0
    
    def update(self, elbow_above_shoulder, form_score):
        """
        elbow_above_shoulder: bool - True if elbow Y < shoulder Y (above)
        form_score: float - form quality (1.0 = perfect, lower = worse)
        """
        if elbow_above_shoulder is None:
            return None
        
        now = time.time()
        
        if self.state == "up":
            # User is in up position
            if not elbow_above_shoulder:  # Elbow goes below shoulder
                self.state = "going_down"
                self.frames = 1
        
        elif self.state == "going_down":
            self.frames += 1
            if not elbow_above_shoulder and self.frames >= self.min_frames:
                self.state = "down"
                self.frames = 0
                self.bottom_form_score = form_score  # Capture form at bottom
        
        elif self.state == "down":
            # User is in down position
            if elbow_above_shoulder:  # Elbow goes above shoulder
                self.state = "going_up"
                self.frames = 1
        
        elif self.state == "going_up":
            self.frames += 1
            if elbow_above_shoulder and self.frames >= self.min_frames:
                # Rep completed!
                if now - self.last_rep_time > 0.7:
                    self.last_rep_time = now
                    
                    # Check if it was a good or bad rep based on form at bottom
                    if self.bottom_form_score >= 0.7:
                        self.reps += 1
                        self.last_rep_feedback = "✅ Good pushup"
                        speak_async("Good pushup")
                    else:
                        self.bad_reps += 1
                        self.last_rep_feedback = "❌ Bad form - elbows too wide"
                        speak_async("Bad form, bring elbows in")
                    
                    self.feedback_timer = now
                
                self.state = "up"
                self.frames = 0
        
        return None
    
    def reset_state_only(self):
        """Reset state machine but keep rep counts"""
        self.state = "up"
        self.frames = 0
        self.last_rep_feedback = ""
        self.bottom_form_score = 1.0
    
    def reset(self):
        """Full reset including rep counts"""
        self.state = "up"
        self.frames = 0
        self.reps = 0
        self.bad_reps = 0
        self.last_rep_feedback = ""
        self.bottom_form_score = 1.0
# Instantiate the new pushup counter
PUSHUP_COUNTER = PushupCounter()
# ---------- Keep old generic counter for compatibility ----------
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
                if now - self.last_rep_time > 0.7: 
                    self.state, self.frames = "top", 0
                    self.reps += 1
                    self.last_rep_time = now
                    tr = "rep"
        return tr
    
    def reset(self):
        self.state, self.frames, self.reps, self.bad_reps = "top", 0, 0, 0
        self.last_feedback = ""
# ---------- Smoothing ----------
SMOOTH = {"elbow_ratio": deque(maxlen=5)}
def smooth_metric(name,val):
    if val is None: return None
    q = SMOOTH[name]; q.append(float(val))
    return float(np.mean(q))
# --- ANALYSIS: PUSHUP - Returns elbow position and form score ---
def analyze_pushup(lms, w, h):
    """
    Returns: (elbow_above_shoulder, form_score, shoulder_elbow_dist, elbow_ratio)
    """
    # Get landmarks
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    re = get(lms, L.RIGHT_ELBOW, w, h)
    ls = get(lms, L.LEFT_SHOULDER, w, h)
    
    # Check visibility
    if not ok_vis(rs[2], re[2], ls[2]):
        return None, 1.0, None, None
    
    # Calculate horizontal distance between right shoulder and right elbow
    shoulder_elbow_dist = abs(rs[0] - re[0])
    
    # Calculate shoulder width
    shoulder_width = abs(ls[0] - rs[0])
    
    # Calculate ratio
    half_shoulder_width = shoulder_width * 0.5
    elbow_ratio = shoulder_elbow_dist / half_shoulder_width if half_shoulder_width > 0 else 0
    
    # Check if elbow is above shoulder (Y coordinate - lower Y = higher on screen)
    elbow_above_shoulder = re[1] < rs[1]
    
    # Form score based on elbow ratio
    form_score = 1.0
    if elbow_ratio > 1.0:
        form_score = 0.6  # Bad form
    
    return elbow_above_shoulder, form_score, shoulder_elbow_dist, elbow_ratio
# Draw pushup distance visualization
def draw_pushup_distance(view, lms, w, h, shoulder_elbow_dist, elbow_ratio):
    """Draw visual markers showing shoulder and elbow positions."""
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    re = get(lms, L.RIGHT_ELBOW, w, h)
    ls = get(lms, L.LEFT_SHOULDER, w, h)
    
    if not ok_vis(rs[2], re[2], ls[2]):
        return
    
    # Draw the points
    cv2.circle(view, (int(rs[0]), int(rs[1])), 8, (0, 255, 0), -1)
    cv2.circle(view, (int(re[0]), int(re[1])), 8, (0, 0, 255), -1)
    cv2.circle(view, (int(ls[0]), int(ls[1])), 8, (0, 255, 255), -1)
    
    # Draw horizontal line from right shoulder to right elbow
    cv2.line(view, (int(rs[0]), int(rs[1])), (int(re[0]), int(re[1])), (255, 0, 255), 3)
    
    # Draw line between shoulders
    cv2.line(view, (int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1])), (255, 255, 0), 2)
    
    # Display distance text
    dist_text = f"{int(shoulder_elbow_dist)}px"
    text_pos = (int((rs[0] + re[0]) / 2), int(rs[1]) - 30)
    text_color = (0, 255, 0) if elbow_ratio <= 1.0 else (0, 0, 255)
    
    cv2.putText(view, dist_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5)
    cv2.putText(view, dist_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
# ---------- SQUAT CODE ----------
def analyze_squat_depth_and_ratio(lms, w, h):
    rh = get(lms, L.RIGHT_HIP, w, h)
    lh = get(lms, L.LEFT_HIP, w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    la = get(lms, L.LEFT_ANKLE, w, h)
    rk = get(lms, L.RIGHT_KNEE, w, h)
    lk = get(lms, L.LEFT_KNEE, w, h)
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    ls = get(lms, L.LEFT_SHOULDER, w, h)
    
    hip_y = knee_y = depth = ratio = None
    
    if ok_vis(rh[2], lh[2], ra[2], la[2]):
        hip_y  = (rh[1] + lh[1]) / 2.0
        ankle_y = (ra[1] + la[1]) / 2.0
        depth  = ankle_y - hip_y
    
    if ok_vis(rk[2], lk[2]):
        knee_y = (rk[1] + lk[1]) / 2.0
    
    if ok_vis(ls[2], rs[2], lk[2], rk[2]):
        shoulder_dist = np.linalg.norm(np.array(ls[:2]) - np.array(rs[:2]))
        knee_dist = np.linalg.norm(np.array(lk[:2]) - np.array(rk[:2]))
        ratio = (knee_dist / shoulder_dist) if shoulder_dist != 0 else None
    
    return depth, ratio, hip_y, knee_y
def stance_feedback(ratio):
    if ratio is None:
        return ""
    if ratio < 0.83:
        return "Spread knees out."
    elif ratio > 1.40:
        return "Bring knees in."
    else:
        return "Perfect! Keep going!"
class SquatCounter:
    def __init__(self, min_down=CFG["squat"]["min_down"], max_up=CFG["squat"]["max_up"], min_frames=2):
        self.low, self.high, self.min_frames = min_down, max_up, min_frames
        self.state = "top"
        self.frames = 0
        self.reps = 0
        self.bad_reps = 0
        self.bottom_reached = False
        self.max_hip_y = None
        self.bottom_ratio = None
        self.bottom_feedback = ""
        self.last_rep_feedback = ""
        self.last_feedback_time = 0
        self.last_rep_time = 0
    
    def reset_state_only(self):
        """Reset state machine but keep rep counts"""
        self.state = "top"
        self.frames = 0
        self.bottom_reached = False
        self.max_hip_y = None
        self.bottom_ratio = None
        self.bottom_feedback = ""
        self.last_rep_feedback = ""
    
    def reset(self):
        """Full reset including rep counts"""
        self.__init__(self.low, self.high, self.min_frames)
    
    def update(self, depth, hip_y, knee_y, ratio):
        if depth is None or hip_y is None or knee_y is None:
            return None
        
        now = time.time()
        below_knee = (hip_y > knee_y)
        
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
SQUAT_COUNTER = SquatCounter()
# ---------- ML prediction ----------
EX_BUF = deque(maxlen=12)
def predict_exercise_ml(lms):
    if not USE_ML or EX_MODEL is None: return "pushup"  # Changed default to pushup
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
# ---------- Main ----------
def main():
    print("Keys: s=start set, e=end set (stats), q=quit")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found.")
        return
    
    current_ex = "pushup"  # Changed default to pushup
    pending_switch = deque(maxlen=8)
    squat_mode_announced = False
    
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
                
                # Check if whole body is visible
                whole_body_visible = is_whole_body_visible(res.pose_landmarks.landmark, w, h)
                
                # Determine exercise mode based on body visibility
                if whole_body_visible:
                    # Switch to squat mode if whole body is visible
                    if current_ex != "squat":
                        current_ex = "squat"
                        PUSHUP_COUNTER.reset_state_only()  # Only reset state, keep counts
                        SQUAT_COUNTER.reset_state_only()  # Only reset state, keep counts
                        if not squat_mode_announced:
                            speak_async("Squat mode activated")
                            squat_mode_announced = True
                else:
                    # Default to pushup mode
                    if current_ex != "pushup":
                        current_ex = "pushup"
                        PUSHUP_COUNTER.reset_state_only()  # Only reset state, keep counts
                        SQUAT_COUNTER.reset_state_only()  # Only reset state, keep counts
                        squat_mode_announced = False
                
                # Distance info (still calculated but not displayed)
                distance = estimate_camera_distance(res.pose_landmarks.landmark, w, h)
                knees_px = knee_distance(res.pose_landmarks.landmark, w, h)
                ideal_knee_min = 60 * (distance / 0.9)
                ideal_knee_max = 70 * (distance / 0.9)
                
                # Removed distance display
                if knees_px:
                    draw_text(view, f"KneeDist: {int(knees_px)} px (ideal {int(ideal_knee_min)}–{int(ideal_knee_max)})",
                             (10, 100), (120,50,220), 0.7, 2)
                
                # ---------- PUSHUP PATH ----------
                if current_ex == "pushup":
                    elbow_above_shoulder, form_score, shoulder_elbow_dist, elbow_ratio = analyze_pushup(
                        res.pose_landmarks.landmark, w, h
                    )
                    
                    # Smooth the elbow ratio for display
                    if elbow_ratio is not None:
                        elbow_ratio_smooth = smooth_metric("elbow_ratio", elbow_ratio)
                    else:
                        elbow_ratio_smooth = None
                    
                    # Update pushup counter
                    PUSHUP_COUNTER.update(elbow_above_shoulder, form_score)
                    
                    # Display visual markers
                    if shoulder_elbow_dist is not None and elbow_ratio is not None:
                        draw_pushup_distance(view, res.pose_landmarks.landmark, w, h, shoulder_elbow_dist, elbow_ratio)
                        
                        ratio_color = (0, 255, 0) if elbow_ratio <= 1.0 else (0, 0, 255)
                        draw_text(view, f"Shoulder-Elbow Distance: {int(shoulder_elbow_dist)}px", (10, 130), ratio_color, 1.0, 2)
                        draw_text(view, f"Elbow Ratio: {elbow_ratio:.2f} (max 1.0)", (10, 160), ratio_color, 0.9, 2)
                    
                    # Display feedback after rep completion
                    if time.time() - PUSHUP_COUNTER.feedback_timer < 2 and PUSHUP_COUNTER.last_rep_feedback:
                        good = PUSHUP_COUNTER.last_rep_feedback.startswith("✅")
                        feedback_color = (0, 200, 0) if good else (0, 0, 255)
                        draw_text(view, PUSHUP_COUNTER.last_rep_feedback, (10, 190), feedback_color, 1.0, 2)
                
                # ---------- SQUAT PATH ----------
                elif current_ex == "squat":
                    depth, ratio, hip_y, knee_y = analyze_squat_depth_and_ratio(
                        res.pose_landmarks.landmark, w, h
                    )
                    if ratio is not None:
                        fb = stance_feedback(ratio)
                        color = (0, 200, 0) if "Perfect" in fb else (0, 0, 255) if "Bring" in fb else (255, 200, 0)
                        draw_text(view, f"Knee/Shoulder Ratio: {ratio:.2f}", (10, 130), color, 0.8, 2)
                        
                    SQUAT_COUNTER.update(depth, hip_y, knee_y, ratio)
                    if time.time() - SQUAT_COUNTER.last_feedback_time < 2 and SQUAT_COUNTER.last_rep_feedback:
                        good = SQUAT_COUNTER.last_rep_feedback.startswith("✅")
                        draw_text(view, SQUAT_COUNTER.last_rep_feedback, (10, 160),
                                 (0,200,0) if good else (0,0,255), 0.8, 2)
            
            # HUD - Display rep counts
            draw_text(view,f"Exercise: {current_ex.upper()}",(10,30),(10,10,10),0.9,2)
            draw_text(view,f"Pushups: Good={PUSHUP_COUNTER.reps} Bad={PUSHUP_COUNTER.bad_reps}",(10,65),(30,30,30),0.85,2)
            draw_text(view,f"Squats: Good={SQUAT_COUNTER.reps} Bad={SQUAT_COUNTER.bad_reps}",
                      (10,90),(30,30,30),0.85,2)
            
            cv2.imshow("Form Tracker", view)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                for q in SMOOTH.values(): q.clear()
                PUSHUP_COUNTER.reset()
                SQUAT_COUNTER.reset()
                speak_async("Starting new set")
            elif key == ord('e'):
                speak_async("Workout ended. Showing statistics.")
                print("\n===== Session Statistics =====")
                pu_total = PUSHUP_COUNTER.reps + PUSHUP_COUNTER.bad_reps
                pu_acc = (PUSHUP_COUNTER.reps / pu_total * 100) if pu_total>0 else 0
                print(f"PUSHUP: Good reps={PUSHUP_COUNTER.reps}, Bad reps={PUSHUP_COUNTER.bad_reps}, Accuracy={pu_acc:.1f}%")
                sq_total = SQUAT_COUNTER.reps + SQUAT_COUNTER.bad_reps
                sq_acc = (SQUAT_COUNTER.reps / sq_total * 100) if sq_total>0 else 0
                print(f"SQUAT : Good reps={SQUAT_COUNTER.reps}, Bad reps={SQUAT_COUNTER.bad_reps}, Accuracy={sq_acc:.1f}%")
                print("==============================\n")
    
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()