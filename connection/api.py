from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import time
import base64
import json
from collections import deque
import asyncio
import warnings
import os
import sys

# Import database
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import SessionLocal, WorkoutSession, WorkoutSet

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def to_np(pt): 
    return np.array(pt[:2], dtype=np.float32)

def angle_3pt(a, b, c):
    a, b, c = to_np(a), to_np(b), to_np(c)
    ab, cb = a - b, c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(ang if ang <= 180 else 360 - ang)

def get(lms, idx, w, h):
    lm = lms[idx.value]
    return (lm.x * w, lm.y * h, lm.visibility)

def ok_vis(*vals, t=0.5):
    return all(v is not None and v >= t for v in vals)

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
    
    la = get(lms, L.LEFT_ANKLE, w, h)
    ra = get(lms, L.RIGHT_ANKLE, w, h)
    
    if la[1] >= h * 0.95 or ra[1] >= h * 0.95:
        return False
    
    return True

# ---------- NEW: Pushup Counter Based on Elbow Y Position ----------
class PushupCounter:
    def __init__(self, min_frames=2):
        self.min_frames = min_frames
        self.state = "up"
        self.frames = 0
        self.reps = 0
        self.bad_reps = 0
        self.last_rep_time = 0
        self.bottom_form_score = 1.0
        self.last_rep_feedback = ""
        self.feedback_timer = 0
        self.last_sound_event = None
        self.sound_event_time = 0
    
    def update(self, elbow_above_shoulder, form_score):
        if elbow_above_shoulder is None:
            return None
        
        now = time.time()
        
        if self.state == "up":
            if not elbow_above_shoulder:
                self.state = "going_down"
                self.frames = 1
        
        elif self.state == "going_down":
            self.frames += 1
            if not elbow_above_shoulder and self.frames >= self.min_frames:
                self.state = "down"
                self.frames = 0
                self.bottom_form_score = form_score
        
        elif self.state == "down":
            if elbow_above_shoulder:
                self.state = "going_up"
                self.frames = 1
        
        elif self.state == "going_up":
            self.frames += 1
            if elbow_above_shoulder and self.frames >= self.min_frames:
                if now - self.last_rep_time > 0.7:
                    self.last_rep_time = now
                    
                    if self.bottom_form_score >= 0.7:
                        self.reps += 1
                        self.last_rep_feedback = "Good pushup"
                        self.last_sound_event = "good"
                        self.sound_event_time = now
                    else:
                        self.bad_reps += 1
                        self.last_rep_feedback = "Bad form - elbows too wide"
                        self.last_sound_event = "bad"
                        self.sound_event_time = now
                    
                    self.feedback_timer = now
                
                self.state = "up"
                self.frames = 0
        
        return None
    
    def reset(self):
        self.state = "up"
        self.frames = 0
        self.reps = 0
        self.bad_reps = 0
        self.last_rep_feedback = ""
        self.bottom_form_score = 1.0
        self.last_sound_event = None

# ---------- Squat Analysis ----------
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
        hip_y = (rh[1] + lh[1]) / 2.0
        ankle_y = (ra[1] + la[1]) / 2.0
        depth = ankle_y - hip_y
    
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
        return "Spread knees out"
    elif ratio > 1.40:
        return "Bring knees in"
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
        self.last_sound_event = None
        self.sound_event_time = 0
    
    def reset(self):
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
                        self.last_rep_feedback = "Good squat"
                        self.last_sound_event = "good"
                        self.sound_event_time = now
                    else:
                        self.bad_reps += 1
                        fb = self.bottom_feedback if self.bottom_feedback else "Adjust your stance"
                        self.last_rep_feedback = fb
                        self.last_sound_event = "bad"
                        self.sound_event_time = now
                    
                    self.last_feedback_time = now
                self.state = "top"
                self.frames = 0

# ---------- Pushup Analysis ----------
def analyze_pushup(lms, w, h):
    rs = get(lms, L.RIGHT_SHOULDER, w, h)
    re = get(lms, L.RIGHT_ELBOW, w, h)
    ls = get(lms, L.LEFT_SHOULDER, w, h)
    
    if not ok_vis(rs[2], re[2], ls[2]):
        return None, 1.0, None, None
    
    shoulder_elbow_dist = abs(rs[0] - re[0])
    shoulder_width = abs(ls[0] - rs[0])
    half_shoulder_width = shoulder_width * 0.5
    elbow_ratio = shoulder_elbow_dist / half_shoulder_width if half_shoulder_width > 0 else 0
    
    elbow_above_shoulder = re[1] < rs[1]
    
    form_score = 1.0
    if elbow_ratio > 1.0:
        form_score = 0.6
    
    return elbow_above_shoulder, form_score, shoulder_elbow_dist, elbow_ratio

# Create global counters
PUSHUP_COUNTER = PushupCounter()
SQUAT_COUNTER = SquatCounter()

# ---------- Smoothing ----------
SMOOTH = {"elbow_ratio": deque(maxlen=5)}

def smooth_metric(name, val):
    if val is None:
        return None
    q = SMOOTH[name]
    q.append(float(val))
    return float(np.mean(q))

# ---------- ML prediction ----------
EX_BUF = deque(maxlen=12)

def predict_exercise_ml(lms):
    if not USE_ML or EX_MODEL is None:
        return "pushup"
    row = [val for lm in lms for val in (lm.x, lm.y)]
    lab = EX_MODEL.predict([row])[0]
    lab = str(lab).lower().strip()
    if lab.endswith("s"):
        lab = lab[:-1]
    mapping = {"pushups": "pushup", "squats": "squat"}
    lab = mapping.get(lab, lab)
    EX_BUF.append(lab)
    labs, cnts = np.unique(list(EX_BUF), return_counts=True)
    return labs[np.argmax(cnts)]

# ---------- Database Routes ----------
@app.post("/api/save-workout")
async def save_workout(data: dict):
    try:
        db = SessionLocal()
        new_session = WorkoutSession()
        db.add(new_session)
        db.flush()
        
        for exercise, stats in data.get("exercises", {}).items():
            new_set = WorkoutSet(
                session_id=new_session.id,
                exercise=exercise,
                good_reps=stats.get("good_reps", 0),
                bad_reps=stats.get("bad_reps", 0)
            )
            db.add(new_set)
        
        db.commit()
        session_id = new_session.id
        db.close()
        
        return {"status": "success", "session_id": session_id, "message": "Workout saved!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/workout-history")
async def get_workout_history():
    try:
        db = SessionLocal()
        sessions = db.query(WorkoutSession).order_by(WorkoutSession.timestamp.desc()).all()
        
        result = []
        for session in sessions:
            session_data = {
                "id": session.id,
                "timestamp": session.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "exercises": []
            }
            for workout_set in session.sets:
                session_data["exercises"].append({
                    "exercise": workout_set.exercise,
                    "good_reps": workout_set.good_reps,
                    "bad_reps": workout_set.bad_reps
                })
            result.append(session_data)
        
        db.close()
        return {"status": "success", "sessions": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/delete-workout/{session_id}")
async def delete_workout(session_id: int):
    try:
        db = SessionLocal()
        session = db.query(WorkoutSession).filter(WorkoutSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            db.close()
            return {"status": "success", "message": "Workout deleted"}
        else:
            db.close()
            return {"status": "error", "message": "Workout not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------- WebSocket Endpoint ----------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Camera not found"})
        return
    
    current_ex = "pushup"
    pending_switch = deque(maxlen=8)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                
                notes = []
                rep_event = None
                sound_event = None
                
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    whole_body_visible = is_whole_body_visible(res.pose_landmarks.landmark, w, h)
                    
                    if whole_body_visible:
                        if current_ex != "squat":
                            current_ex = "squat"
                    else:
                        if current_ex != "pushup":
                            current_ex = "pushup"
                    
                    # PUSHUP PATH
                    if current_ex == "pushup":
                        elbow_above_shoulder, form_score, shoulder_elbow_dist, elbow_ratio = analyze_pushup(
                            res.pose_landmarks.landmark, w, h
                        )
                        
                        PUSHUP_COUNTER.update(elbow_above_shoulder, form_score)
                        
                        # Add form feedback to notes
                        if elbow_ratio is not None and elbow_ratio > 1.0:
                            notes.append("Elbows too wide - bring them closer to your body")
                        
                        # Check for rep event
                        if time.time() - PUSHUP_COUNTER.feedback_timer < 1.0:
                            if "Good" in PUSHUP_COUNTER.last_rep_feedback:
                                rep_event = "good"
                            else:
                                rep_event = "bad"
                                notes.append(PUSHUP_COUNTER.last_rep_feedback)
                        
                        # Check for sound event
                        if time.time() - PUSHUP_COUNTER.sound_event_time < 0.5:
                            sound_event = PUSHUP_COUNTER.last_sound_event
                            PUSHUP_COUNTER.last_sound_event = None
                    
                    # SQUAT PATH
                    elif current_ex == "squat":
                        depth, ratio, hip_y, knee_y = analyze_squat_depth_and_ratio(
                            res.pose_landmarks.landmark, w, h
                        )
                        
                        SQUAT_COUNTER.update(depth, hip_y, knee_y, ratio)
                        
                        # Add stance feedback to notes
                        if ratio is not None:
                            fb = stance_feedback(ratio)
                            if "Perfect" not in fb:
                                notes.append(fb)
                        
                        # Check for rep event
                        if time.time() - SQUAT_COUNTER.last_feedback_time < 1.0:
                            if "Good" in SQUAT_COUNTER.last_rep_feedback:
                                rep_event = "good"
                            else:
                                rep_event = "bad"
                                notes.append(SQUAT_COUNTER.last_rep_feedback)
                        
                        # Check for sound event
                        if time.time() - SQUAT_COUNTER.sound_event_time < 0.5:
                            sound_event = SQUAT_COUNTER.last_sound_event
                            SQUAT_COUNTER.last_sound_event = None
                
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                data = {
                    "frame": frame_base64,
                    "exercise": current_ex,
                    "reps": {
                        "pushup": PUSHUP_COUNTER.reps,
                        "squat": SQUAT_COUNTER.reps
                    },
                    "good_reps": {
                        "pushup": PUSHUP_COUNTER.reps,
                        "squat": SQUAT_COUNTER.reps
                    },
                    "bad_reps": {
                        "pushup": PUSHUP_COUNTER.bad_reps,
                        "squat": SQUAT_COUNTER.bad_reps
                    },
                    "notes": notes,
                    "rep_event": rep_event,
                    "sound_event": sound_event
                }
                
                await websocket.send_json(data)
                
                try:
                    command = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    cmd_data = json.loads(command)
                    
                    if cmd_data.get("action") == "change_exercise":
                        current_ex = cmd_data.get("exercise", "pushup")
                        pending_switch.clear()
                    elif cmd_data.get("action") == "reset":
                        PUSHUP_COUNTER.reset()
                        SQUAT_COUNTER.reset()
                        for q in SMOOTH.values():
                            q.clear()
                    elif cmd_data.get("action") == "save_workout":
                        db = SessionLocal()
                        new_session = WorkoutSession()
                        db.add(new_session)
                        db.flush()
                        
                        if PUSHUP_COUNTER.reps > 0 or PUSHUP_COUNTER.bad_reps > 0:
                            new_set = WorkoutSet(
                                session_id=new_session.id,
                                exercise="pushup",
                                good_reps=PUSHUP_COUNTER.reps,
                                bad_reps=PUSHUP_COUNTER.bad_reps
                            )
                            db.add(new_set)
                        
                        if SQUAT_COUNTER.reps > 0 or SQUAT_COUNTER.bad_reps > 0:
                            new_set = WorkoutSet(
                                session_id=new_session.id,
                                exercise="squat",
                                good_reps=SQUAT_COUNTER.reps,
                                bad_reps=SQUAT_COUNTER.bad_reps
                            )
                            db.add(new_set)
                        
                        db.commit()
                        db.close()
                        
                        await websocket.send_json({
                            "action": "workout_saved",
                            "message": "Workout saved to database!"
                        })
                        
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.03)
                
        except WebSocketDisconnect:
            print("Client disconnected")
        finally:
            cap.release()

@app.get("/")
async def get_html():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        paths = [
            os.path.join(base_dir, "..", "frontend", "website.html"),
            os.path.join(base_dir, "frontend", "website.html"),
            "frontend/website.html",
            "../frontend/website.html"
        ]
        
        for path in paths:
            if os.path.exists(path):
                print(f"✅ Found HTML at: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    return HTMLResponse(
                        content=content,
                        headers={
                            "Cache-Control": "no-cache, no-store, must-revalidate",
                            "Pragma": "no-cache",
                            "Expires": "0"
                        }
                    )
        
        error_msg = f"""
        <h1>❌ website.html not found!</h1>
        <p>Searched in:</p>
        <ul>
            {"".join(f"<li>{p}</li>" for p in paths)}
        </ul>
        """
        return HTMLResponse(content=error_msg, status_code=404)
        
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Fitness Coach API is running",
        "ml_enabled": USE_ML,
        "model_loaded": EX_MODEL is not None,
        "database": "connected"
    }

# To run the server, use the command:
#cd C:\Users\User\.vscode\shellhacks\multi_tool_agent\opencv_fitness_coach
#uvicorn connection.api:app --reload