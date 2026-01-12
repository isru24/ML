import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("mental_health_dataset.csv")

df = df.dropna()

cat_cols = ['gender', 'employment_status', 'work_environment', 'mental_health_history', 'seeks_treatment']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

features = [
    'age', 'gender', 'employment_status', 'work_environment',
    'mental_health_history', 'seeks_treatment', 'stress_level',
    'sleep_hours', 'physical_activity_days', 'depression_score',
    'anxiety_score', 'social_support_score', 'productivity_score'
]

target = 'mental_health_risk'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

print(" Getting Things Ready!")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n Model performance:\n", classification_report(y_test, y_pred))

joblib.dump(model, "mental_health_risk_model.pkl")
print("Model saved as mental_health_risk_model.pkl")


def generate_supportive_response(risk_level):
    risk_level = risk_level.lower()
    if "low" in risk_level:
        return "You're doing well! Keep maintaining your healthy habits and self-care routines."
    elif "medium" in risk_level:
        return "You seem to be under moderate stress. Take some time for yourself — small breaks and social support can make a big difference."
    elif "high" in risk_level:
        return "It looks like you might be at higher risk. Please consider reaching out to a mental health professional or someone you trust — you don’t have to face this alone."
    else:
        return "I'm here to support you. Tell me more about how you've been feeling."

print("\n💬 Supportive AI is ready! Type your info below (or 'quit' to stop):\n")

while True:
    user_input = input("Do you want to assess your mental health risk? (yes/quit): ").strip().lower()
    if user_input == "quit":
        print("👋 Take care — remember, you matter and help is always available.")
        break

    try:
        age = float(input("Age: "))
        if age < 18:
            print("Sorry you are underage for this program")
            break
        stress = float(input("Stress level (0–100): "))
        sleep = float(input("Sleep hours (per night): "))
        depression = float(input("Depression score (0–100): "))
        anxiety = float(input("Anxiety score (0–100): "))
        social = float(input("Social support score (0–100): "))
        productivity = float(input("Productivity score (0–100): "))

        sample = pd.DataFrame([{
            'age': age,
            'gender': 1,
            'employment_status': 1,
            'work_environment': 1,
            'mental_health_history': 1,
            'seeks_treatment': 1,
            'stress_level': stress,
            'sleep_hours': sleep,
            'physical_activity_days': 3,
            'depression_score': depression,
            'anxiety_score': anxiety,
            'social_support_score': social,
            'productivity_score': productivity
        }])

        risk = model.predict(sample)[0]
        print(f"\n🧠 Predicted Mental Health Risk: {risk}")
        print(f"💬 AI: {generate_supportive_response(risk)}\n")

    except Exception as e:
        print(f"⚠️ Error: {e}\nPlease make sure to enter valid numbers.\n")
