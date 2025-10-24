from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exercise_type = "pushup"  # default


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    global exercise_type
    exercise_type = request.form.get('exercise')
    return ('', 204)


def analyze_pose(landmarks, image):
    """Analyze posture based on selected exercise and give feedback."""
    h, w, _ = image.shape
    feedback = "Good form!"

    def angle(a, b, c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    if exercise_type == "pushup":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        arm_angle = angle(shoulder, elbow, wrist)
        if arm_angle > 160:
            feedback = "Lower your body more"
        elif arm_angle < 70:
            feedback = "Push back up"
        else:
            feedback = "Good push-up!"

    elif exercise_type == "squat":
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        knee_angle = angle(hip, knee, ankle)
        if knee_angle > 160:
            feedback = "Go lower!"
        elif knee_angle < 70:
            feedback = "Stand up!"
        else:
            feedback = "Nice squat!"

    elif exercise_type == "plank":
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        body_angle = angle(shoulder, hip, ankle)
        if body_angle < 160:
            feedback = "Keep your body straight!"
        else:
            feedback = "Good plank!"

    elif exercise_type == "pullup":
        wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        if wrist_y < shoulder_y:
            feedback = "Lower yourself down"
        else:
            feedback = "Pull up!"

    return feedback


def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            feedback = "Waiting for pose..."

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                feedback = analyze_pose(results.pose_landmarks.landmark, frame)

            cv2.putText(frame, f"{exercise_type.upper()} - {feedback}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
