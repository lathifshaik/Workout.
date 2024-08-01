from app import db, WorkoutSummary
from datetime import datetime

# Create a new summary record
summary = WorkoutSummary(
    user_id=1,  # Replace with an actual user_id
    exercise="PushUps",
    total_duration_seconds=300,
    total_calories_burned=45,
    date=datetime.now().date()
)

# Add and commit to the database
db.session.add(summary)
db.session.commit()
