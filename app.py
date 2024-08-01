from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(minutes=30)
db = SQLAlchemy(app)

# Load your model
model_path = 'LRCN_model___Date_Time_2024_07_12__19_36_49___Loss_0.5270757079124451___Accuracy_0.8085106611251831.h5'
model = tf.keras.models.load_model(model_path)

# List of exercise classes
CLASSES_LIST = ["JumpingJack", "BodyWeightSquats", "Lunges", "PullUps", "PushUps"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)

    def calculate_bmi(self):
        height_in_meters = self.height / 100
        bmi = self.weight / (height_in_meters ** 2)
        return bmi

class WorkoutSummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    exercise = db.Column(db.String(100), nullable=False)
    total_duration_seconds = db.Column(db.Float, nullable=False)
    total_calories_burned = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)

    
def create_db():
    with app.app_context():
        db.create_all()

def calculate_calories_burned(duration_seconds, exercise_type):
    calorie_burn_rates = {
        "JumpingJack": 10,
        "BodyWeightSquats": 8,
        "Lunges": 7,
        "PullUps": 12,
        "PushUps": 9
    }
    rate_per_minute = calorie_burn_rates.get(exercise_type, 8)
    return (duration_seconds / 60) * rate_per_minute

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    return frame_normalized


def generate_predictions():
    cap = cv2.VideoCapture(0)
    frame_sequence = []
    sequence_length = 20
    current_exercise = None
    exercise_start_time = None
    exercise_logs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw skeleton on frame
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        preprocessed_frame = preprocess_frame(frame)
        frame_sequence.append(preprocessed_frame)

        if len(frame_sequence) > sequence_length:
            frame_sequence.pop(0)

        if len(frame_sequence) == sequence_length:
            input_sequence = np.expand_dims(frame_sequence, axis=0)
            prediction = model.predict(input_sequence)
            predicted_class_index = np.argmax(prediction)
            predicted_class = CLASSES_LIST[predicted_class_index]

            if current_exercise and current_exercise != predicted_class:
                end_time = datetime.now()
                duration_seconds = (end_time - exercise_start_time).total_seconds()
                calories_burned = calculate_calories_burned(duration_seconds, current_exercise)
                exercise_logs.append({
                    "exercise": current_exercise,
                    "start_time": exercise_start_time,
                    "end_time": end_time,
                    "calories_burned": calories_burned
                })
                exercise_start_time = datetime.now()

            current_exercise = predicted_class
            if not exercise_start_time:
                exercise_start_time = datetime.now()

            cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    if 'user_id' in session:
        user_id = session['user_id']
        for log in exercise_logs:
            new_log = WorkoutLog(
                user_id=user_id,
                exercise=log['exercise'],
                start_time=log['start_time'],
                end_time=log['end_time'],
                calories_burned=log['calories_burned']
            )
            db.session.add(new_log)

            # Update the WorkoutSummary table
            existing_summary = WorkoutSummary.query.filter_by(
                user_id=user_id,
                exercise=log['exercise'],
                date=datetime.now().date()
            ).first()

            if existing_summary:
                existing_summary.total_duration_seconds += (log['end_time'] - log['start_time']).total_seconds()
                existing_summary.total_calories_burned += log['calories_burned']
            else:
                new_summary = WorkoutSummary(
                    user_id=user_id,
                    exercise=log['exercise'],
                    total_duration_seconds=(log['end_time'] - log['start_time']).total_seconds(),
                    total_calories_burned=log['calories_burned'],
                    date=datetime.now().date()
                )
                db.session.add(new_summary)

        db.session.commit()




@app.route('/video_feed')
def video_feed():
    return Response(generate_predictions(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        weight = request.form['weight']
        height = request.form['height']
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different username.', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password, age=int(age),
                        weight=float(weight), height=float(height))
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session.permanent = True
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    bmi = user.calculate_bmi() if user else None
    bmi_category = get_bmi_category(bmi) if bmi else None
    return render_template('home.html', user=user, bmi=bmi, bmi_category=bmi_category)

def get_bmi_category(bmi):
    if bmi is None:
        return "BMI not calculated"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal Weight"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

@app.route('/beginner')
def beginner():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    return render_template('beginner.html', user=user, title="Beginner Workouts")

@app.route('/intermediate')
def intermediate():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    return render_template('intermediate.html', user=user, title="Intermediate Workouts")

@app.route('/advanced')
def advanced():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    return render_template('advanced.html', user=user, title="Advanced Workouts")

@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        weight = request.form['weight']
        height = request.form['height']
        
        if weight:
            user.weight = float(weight)
        if height:
            user.height = float(height)
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('home'))
    
    return render_template('update_profile.html', user=user)

@app.route('/workout_summary')
def workout_summary():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    summaries = WorkoutSummary.query.filter_by(user_id=user_id).all()
    
    # Debugging output
    for summary in summaries:
        print(f"Exercise: {summary.exercise}, Duration: {summary.total_duration_seconds}, Calories: {summary.total_calories_burned}, Date: {summary.date}")

    return render_template('workout_summary.html', summaries=summaries)




if __name__ == '__main__':
    create_db()  # Create the database tables
    app.run(debug=True)
