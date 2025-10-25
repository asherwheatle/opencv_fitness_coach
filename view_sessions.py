from database import SessionLocal, WorkoutSession

def show_sessions():
    db = SessionLocal()
    sessions = db.query(WorkoutSession).all()

    for session in sessions:
        print(f"\n=== Session {session.id} ({session.timestamp.strftime('%Y-%m-%d %H:%M')}) ===")
        for s in session.sets:
            print(f"  {s.exercise.title()}: {s.good_reps} good, {s.bad_reps} bad")
    db.close()

if __name__ == "__main__":
    show_sessions()