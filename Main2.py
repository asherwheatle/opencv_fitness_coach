import cv2
import numpy as np
import mediapipe as mp
import time, os, warnings, threading
from collections import deque
import pyttsx3
import joblib
from database import SessionLocal, WorkoutSession, WorkoutSet
import tkinter as tk
from tkinter import ttk



# ---------- TEXT TO SPEECH ----------
engine = pyttsx3.init()
engine.setProperty('rate', 185)
engine.setProperty('volume', 1.0)

def speak_async(text):
    """Speak text asynchronously (non-blocking)"""
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
    "pushup": {"nice_elbow_min": 80, "nice_elbow_max": 150},
    "squat":  {"nice_knee_min": 65, "nice_knee_max": 140},
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
def dist(a,b): return float(np.linalg.norm(to_np(a)-to_np(b)))
def get(lms,idx,w,h): lm=lms[idx.value]; return (lm.x*w,lm.y*h,lm.visibility)
def ok_vis(*vals,t=0.5): return all(v is not None and v>=t for v in vals)

# ---------- Rep Counter ----------
class RepCounter:
    def __init__(self, low, high, min_frames=2):
        self.low, self.high, self.min_frames = low, high, min_frames
        self.state, self.frames, self.reps = "top", 0, 0
        self.last_rep_time = 0
        self.bad_reps = 0
        self.warnings = []

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
        self.state, self.frames, self.reps = "top", 0, 0
        self.last_rep_time = 0
        self.bad_reps = 0
        self.warnings.clear()

COUNTERS = {
    "pushup": RepCounter(low=70, high=155),
    "squat":  RepCounter(low=70, high=160)
}

# ---------- Smoothing ----------
SMOOTH = {k: deque(maxlen=5) for k in COUNTERS}
def smooth_metric(name,val):
    if val is None: return None
    q = SMOOTH[name]; q.append(float(val))
    return float(np.mean(q))

# ---------- Analysis ----------
def analyze_pushup(lms,w,h,notes):
    ls,le,lw = get(lms,L.LEFT_SHOULDER,w,h),get(lms,L.LEFT_ELBOW,w,h),get(lms,L.LEFT_WRIST,w,h)
    rs,re,rw = get(lms,L.RIGHT_SHOULDER,w,h),get(lms,L.RIGHT_ELBOW,w,h),get(lms,L.RIGHT_WRIST,w,h)
    okL,okR = ok_vis(ls[2],le[2],lw[2]), ok_vis(rs[2],re[2],rw[2])
    elbows=[]
    if okL: elbows.append(angle_3pt(ls,le,lw))
    if okR: elbows.append(angle_3pt(rs,re,rw))
    if not elbows: return None,0.0
    elbow_mean=np.mean(elbows)
    form_score=1.0
    if elbow_mean < CFG["pushup"]["nice_elbow_min"] or elbow_mean > CFG["pushup"]["nice_elbow_max"]:
        form_score-=0.2
        msg="Adjust arm range — spread out arms"
        notes.append(msg)
        speak_async(msg)
    return elbow_mean, form_score

def analyze_squat(lms,w,h,notes):
    lh,lk,la = get(lms,L.LEFT_HIP,w,h),get(lms,L.LEFT_KNEE,w,h),get(lms,L.LEFT_ANKLE,w,h)
    rh,rk,ra = get(lms,L.RIGHT_HIP,w,h),get(lms,L.RIGHT_KNEE,w,h),get(lms,L.RIGHT_ANKLE,w,h)

    if not ok_vis(lh[2], lk[2], la[2], rh[2], rk[2], ra[2], t=0.5):
        msg="Lower body not fully visible — move back slightly"
        notes.append(msg)
        speak_async(msg)
        return None, None, "pushup"

    knees=[]
    if ok_vis(lh[2],lk[2],la[2]): knees.append(angle_3pt(lh,lk,la))
    if ok_vis(rh[2],rk[2],ra[2]): knees.append(angle_3pt(rh,rk,ra))
    if not knees: return None,0.0,"squat"
    knee_mean=np.mean(knees)
    form_score=1.0
    if knee_mean < CFG["squat"]["nice_knee_min"]:
        form_score-=0.2
        msg="Go deeper"
        notes.append(msg)
        speak_async(msg)
    elif knee_mean > CFG["squat"]["nice_knee_max"]:
        form_score-=0.1
        msg="Stand tall at the top"
        notes.append(msg)
        speak_async(msg)
    return knee_mean, form_score, "squat"


ANALYZE={
    "pushup": analyze_pushup,
    "squat":  analyze_squat,
}

# ---------- ML prediction ----------
EX_BUF=deque(maxlen=12)

def predict_exercise_ml(lms):
    if not USE_ML or EX_MODEL is None: return "push-up"
    row=[val for lm in lms for val in (lm.x,lm.y)]
    lab=EX_MODEL.predict([row])[0]
    lab=str(lab).lower().strip()
    if lab.endswith("s"): lab=lab[:-1]
    mapping={"pushups":"pushup","squats":"squat"}
    lab=mapping.get(lab,lab)
    EX_BUF.append(lab)
    labs,cnts=np.unique(list(EX_BUF),return_counts=True)
    return labs[np.argmax(cnts)]

# ---------- Helper for readable dark text ----------
def draw_text(view, text, pos, color=(10,10,10), scale=0.8, thick=2):
    x, y = pos
    cv2.putText(view, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick+3)
    cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ---------- Past Sessions Viewer ----------
def show_past_sessions():
    """Show past workout sessions in a Tkinter window"""
    db = SessionLocal()
    sessions = db.query(WorkoutSession).order_by(WorkoutSession.timestamp.desc()).all()
    db.close()
    
    if not sessions:
        # Show message if no sessions
        root = tk.Tk()
        root.title("Workout History")
        root.geometry("400x200")
        
        label = tk.Label(root, text="No workout sessions found.", font=("Arial", 12))
        label.pack(expand=True)
        
        close_btn = tk.Button(root, text="Close", command=root.destroy)
        close_btn.pack(pady=10)
        
        root.mainloop()
        return
    
    # Create main window
    root = tk.Tk()
    root.title("Workout History")
    root.geometry("800x600")
    
    # Create frame for treeview and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create treeview
    tree = ttk.Treeview(frame, columns=("Session", "Date", "Exercise", "Good Reps", "Bad Reps"), show="headings")
    
    # Configure columns
    tree.heading("Session", text="Session ID")
    tree.heading("Date", text="Date & Time")
    tree.heading("Exercise", text="Exercise")
    tree.heading("Good Reps", text="Good Reps")
    tree.heading("Bad Reps", text="Bad Reps")
    
    tree.column("Session", width=80)
    tree.column("Date", width=150)
    tree.column("Exercise", width=120)
    tree.column("Good Reps", width=100)
    tree.column("Bad Reps", width=100)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Pack treeview and scrollbar
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Insert data
    for session in sessions:
        for set_data in session.sets:
            tree.insert("", "end", values=(
                session.id,
                session.timestamp.strftime('%Y-%m-%d %H:%M'),
                set_data.exercise.title(),
                set_data.good_reps,
                set_data.bad_reps
            ))
    
    # Add close button
    close_btn = tk.Button(root, text="Close", command=root.destroy)
    close_btn.pack(pady=10)
    
    root.mainloop()

# ---------- Main ----------
def main():
    print("Keys: s=start set, e=end set, q=quit")
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found."); return
    current_ex="push-up"
    pending_switch=deque(maxlen=8)
    last_feedback=""; feedback_timer=0
    show_summary=False
    summary_text=[]

    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            view=frame.copy()
            h,w=frame.shape[:2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=pose.process(rgb)
            notes=[]
            metric,form_score=None,1.0

            if res.pose_landmarks:
                mp_draw.draw_landmarks(view,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                if USE_ML and EX_MODEL is not None:
                    lab=predict_exercise_ml(res.pose_landmarks.landmark)
                else:
                    lab=current_ex
                # Debug: show if pose is detected
                draw_text(view,"POSE DETECTED",(10,140),(0,255,0),0.6,1)
                if lab!=current_ex:
                    pending_switch.append(lab)
                    if len(pending_switch)==pending_switch.maxlen and len(set(pending_switch))==1:
                        current_ex=lab; pending_switch.clear()

                # Analyze exercise
                if current_ex in ANALYZE:
                    result=ANALYZE[current_ex](res.pose_landmarks.landmark,w,h,notes)
                    if current_ex=="squat":
                        metric_raw,form_score,new_type=result
                        if new_type=="pushup":
                            current_ex="pushup"
                            continue
                    else:
                        metric_raw,form_score=result
                    metric=smooth_metric(current_ex,metric_raw)

                    if current_ex in COUNTERS and metric is not None:
                        tr=COUNTERS[current_ex].update(metric)
                        if tr=="rep":
                            if form_score>=0.8:
                                last_feedback=f"✅ Good {current_ex} rep!"
                                speak_async(f"Good {current_ex}")
                            else:
                                last_feedback=f"❌ Bad {current_ex} rep – not counted!"
                                speak_async(f"Bad {current_ex} form")
                                COUNTERS[current_ex].bad_reps += 1
                                COUNTERS[current_ex].reps -= 1 if COUNTERS[current_ex].reps>0 else 0
                            feedback_timer=time.time()

            # HUD
            draw_text(view,f"Exercise: {current_ex.upper()}",(10,30),(10,10,10),0.9,2)
            reps_str=" | ".join([f"{k}:{COUNTERS[k].reps}" for k in COUNTERS])
            draw_text(view,f"Reps [{reps_str}]",(10,65),(30,30,30),0.85,2)
            
            # Debug info
            if metric is not None:
                draw_text(view,f"Metric: {metric:.1f}",(10,100),(100,100,100),0.6,1)
            if current_ex in COUNTERS:
                state = COUNTERS[current_ex].state
                draw_text(view,f"State: {state}",(10,120),(100,100,100),0.6,1)

            if last_feedback and time.time()-feedback_timer<2:
                color=(0,180,0) if "Good" in last_feedback else (0,0,200)
                draw_text(view,last_feedback,(10,110),color,0.85,2)

            base_y=150
            for i,n in enumerate(notes[:3]):
                draw_text(view,f"• {n}",(10,base_y+25*i),(20,20,20),0.7,2)

            # show summary if requested
            if show_summary:
                draw_text(view,"=== Workout Summary ===",(10,250),(0,0,0),0.8,2)
                for i,line in enumerate(summary_text):
                    draw_text(view,line,(10,280+25*i),(10,10,10),0.75,2)

            cv2.imshow("Form Tracker (Voice Summary)",view)

            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):
                break
            elif key==ord('s'):
                for c in COUNTERS.values(): c.reset()
                for q in SMOOTH.values(): q.clear()
                show_summary=False
                speak_async("Starting new set")
                print("▶️ New set started.")
            elif key == ord('e'):
                show_summary = True
                speak_async("Workout complete. Here is your summary.")

                # --- Database: create new session and sets ---
                db = SessionLocal()
                new_session = WorkoutSession()
                db.add(new_session)
                db.flush()  # ensures new_session.id is available

                table_data = []      # for Tkinter table
                summary_text = []    # optional text summary

                print("\n===== Session Summary =====")
                for ex, c in COUNTERS.items():
                    print(f"DEBUG: {ex} - good_reps: {c.reps}, bad_reps: {c.bad_reps}")
                    # Save each exercise as a WorkoutSet
                    new_set = WorkoutSet(
                        session_id=new_session.id,
                        exercise=ex,
                        good_reps=c.reps,
                        bad_reps=c.bad_reps
                    )
                    db.add(new_set)

                    # Prepare table and text summary
                    table_data.append([ex.title(), c.reps, c.bad_reps])
                    line = f"{ex.upper()}: {c.reps} good reps, {c.bad_reps} bad reps"
                    summary_text.append(line)
                    print(line)
                    if c.bad_reps > 0:
                        speak_async(f"{ex} had {c.bad_reps} bad repetitions")

                # Commit changes
                db.commit()
                print("✅ Workout session saved.")

                # ✅ Capture timestamp BEFORE closing session to avoid DetachedInstanceError
                session_timestamp = new_session.timestamp
                db.close()

                # Reset counters after saving data
                for c in COUNTERS.values(): c.reset()
                for q in SMOOTH.values(): q.clear()

                # --- Tkinter table window ---

                def show_summary_window(session_timestamp, table_data):
                    root = tk.Tk()
                    root.title(f"Workout Summary - {session_timestamp.strftime('%Y-%m-%d %H:%M')}")

                    tree = ttk.Treeview(root, columns=("Exercise", "Good Reps", "Bad Reps"), show="headings")
                    tree.heading("Exercise", text="Exercise")
                    tree.heading("Good Reps", text="Good Reps")
                    tree.heading("Bad Reps", text="Bad Reps")

                    # Insert rows
                    for row in table_data:
                        tree.insert("", tk.END, values=row)

                    tree.pack(expand=True, fill="both")

                    # Instructions label
                    instructions = tk.Label(root, text="Press 'p' to view past sessions, or 'Close' to continue...", 
                                          font=("Arial", 10), fg="blue")
                    instructions.pack(pady=5)

                    # Button frame
                    button_frame = tk.Frame(root)
                    button_frame.pack(pady=5)

                    # Past sessions button
                    past_btn = tk.Button(button_frame, text="View Past Sessions", 
                                       command=lambda: [root.destroy(), show_past_sessions()])
                    past_btn.pack(side=tk.LEFT, padx=5)

                    # Close button
                    close_btn = tk.Button(button_frame, text="Close", command=root.destroy)
                    close_btn.pack(side=tk.LEFT, padx=5)

                    root.mainloop()

                # Show the Tkinter summary window
                show_summary_window(session_timestamp, table_data)
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
