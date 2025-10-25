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
    "pushup": {"nice_elbow_min": 80, "nice_elbow_max": 150},
    "squat": {"nice_knee_min": 65, "nice_knee_max": 140},
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

# ---------- Rep Counter ----------
class RepCounter:
    def __init__(self, low, high, min_frames=2):
        self.low, self.high, self.min_frames = low, high, min_frames
        self.state, self.frames, self.reps = "top", 0, 0
        self.last_rep_time = 0
        self.bad_reps = 0
        self.good_reps = 0
        self.warnings = []

    def update(self, metric):
        if metric is None:
            return None
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
        self.state, self.frames, self.reps = "top", 0, 0
        self.last_rep_time = 0
        self.bad_reps = 0
        self.good_reps = 0
        self.warnings.clear()

COUNTERS = {
    "pushup": RepCounter(low=70, high=155),
    "squat": RepCounter(low=70, high=160)
}

# ---------- Smoothing ----------
SMOOTH = {k: deque(maxlen=5) for k in COUNTERS}

def smooth_metric(name, val):
    if val is None:
        return None
    q = SMOOTH[name]
    q.append(float(val))
    return float(np.mean(q))

# ---------- Analysis ----------
def analyze_pushup(lms, w, h, notes):
    ls, le, lw = get(lms, L.LEFT_SHOULDER, w, h), get(lms, L.LEFT_ELBOW, w, h), get(lms, L.LEFT_WRIST, w, h)
    rs, re, rw = get(lms, L.RIGHT_SHOULDER, w, h), get(lms, L.RIGHT_ELBOW, w, h), get(lms, L.RIGHT_WRIST, w, h)
    okL, okR = ok_vis(ls[2], le[2], lw[2]), ok_vis(rs[2], re[2], rw[2])
    elbows = []
    if okL:
        elbows.append(angle_3pt(ls, le, lw))
    if okR:
        elbows.append(angle_3pt(rs, re, rw))
    if not elbows:
        return None, 0.0
    elbow_mean = np.mean(elbows)
    form_score = 1.0
    if elbow_mean < CFG["pushup"]["nice_elbow_min"] or elbow_mean > CFG["pushup"]["nice_elbow_max"]:
        form_score -= 0.2
        notes.append("Adjust arm range — spread out arms")
    return elbow_mean, form_score

def analyze_squat(lms, w, h, notes):
    lh, lk, la = get(lms, L.LEFT_HIP, w, h), get(lms, L.LEFT_KNEE, w, h), get(lms, L.LEFT_ANKLE, w, h)
    rh, rk, ra = get(lms, L.RIGHT_HIP, w, h), get(lms, L.RIGHT_KNEE, w, h), get(lms, L.RIGHT_ANKLE, w, h)
    
    if not ok_vis(lh[2], lk[2], la[2], rh[2], rk[2], ra[2], t=0.5):
        notes.append("Lower body not fully visible — move back slightly")
        return None, None, "pushup"
    
    knees = []
    if ok_vis(lh[2], lk[2], la[2]):
        knees.append(angle_3pt(lh, lk, la))
    if ok_vis(rh[2], rk[2], ra[2]):
        knees.append(angle_3pt(rh, rk, ra))
    if not knees:
        return None, 0.0, "squat"
    knee_mean = np.mean(knees)
    form_score = 1.0
    if knee_mean < CFG["squat"]["nice_knee_min"]:
        form_score -= 0.2
        notes.append("Go deeper")
    elif knee_mean > CFG["squat"]["nice_knee_max"]:
        form_score -= 0.1
        notes.append("Stand tall at the top")
    return knee_mean, form_score, "squat"

ANALYZE = {
    "pushup": analyze_pushup,
    "squat": analyze_squat
}

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
    """Save workout session to database"""
    try:
        db = SessionLocal()
        
        # Create new session
        new_session = WorkoutSession()
        db.add(new_session)
        db.flush()
        
        # Add workout sets
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
    """Get all workout sessions"""
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
    """Delete a workout session"""
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
    last_feedback = ""
    
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
                metric, form_score = None, 1.0
                rep_event = None
                
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    if USE_ML and EX_MODEL is not None:
                        lab = predict_exercise_ml(res.pose_landmarks.landmark)
                    else:
                        lab = current_ex
                    
                    if lab != current_ex:
                        pending_switch.append(lab)
                        if len(pending_switch) == pending_switch.maxlen and len(set(pending_switch)) == 1:
                            current_ex = lab
                            pending_switch.clear()
                    
                    if current_ex in ANALYZE:
                        result = ANALYZE[current_ex](res.pose_landmarks.landmark, w, h, notes)
                        if current_ex == "squat":
                            metric_raw, form_score, new_type = result
                            if new_type == "pushup":
                                current_ex = "pushup"
                                continue
                        else:
                            metric_raw, form_score = result
                        metric = smooth_metric(current_ex, metric_raw)
                        
                        if current_ex in COUNTERS and metric is not None:
                            tr = COUNTERS[current_ex].update(metric)
                            if tr == "rep":
                                if form_score >= 0.8:
                                    last_feedback = f"Good {current_ex} rep!"
                                    rep_event = "good"
                                    COUNTERS[current_ex].good_reps += 1
                                else:
                                    last_feedback = f"Bad {current_ex} form!"
                                    rep_event = "bad"
                                    COUNTERS[current_ex].bad_reps += 1
                                    COUNTERS[current_ex].reps -= 1 if COUNTERS[current_ex].reps > 0 else 0
                
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                data = {
                    "frame": frame_base64,
                    "exercise": current_ex,
                    "reps": {k: COUNTERS[k].reps for k in COUNTERS},
                    "good_reps": {k: COUNTERS[k].good_reps for k in COUNTERS},
                    "bad_reps": {k: COUNTERS[k].bad_reps for k in COUNTERS},
                    "feedback": last_feedback,
                    "notes": notes,
                    "form_score": form_score,
                    "rep_event": rep_event,
                    "metric": metric,
                    "state": COUNTERS[current_ex].state if current_ex in COUNTERS else "unknown"
                }
                
                await websocket.send_json(data)
                
                try:
                    command = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    cmd_data = json.loads(command)
                    
                    if cmd_data.get("action") == "change_exercise":
                        current_ex = cmd_data.get("exercise", "pushup")
                        pending_switch.clear()
                    elif cmd_data.get("action") == "reset":
                        for c in COUNTERS.values():
                            c.reset()
                        for q in SMOOTH.values():
                            q.clear()
                        last_feedback = "Reset complete!"
                    elif cmd_data.get("action") == "save_workout":
                        # Save to database
                        db = SessionLocal()
                        new_session = WorkoutSession()
                        db.add(new_session)
                        db.flush()
                        
                        for ex, counter in COUNTERS.items():
                            if counter.good_reps > 0 or counter.bad_reps > 0:
                                new_set = WorkoutSet(
                                    session_id=new_session.id,
                                    exercise=ex,
                                    good_reps=counter.good_reps,
                                    bad_reps=counter.bad_reps
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
                    return HTMLResponse(content=f.read())
        
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