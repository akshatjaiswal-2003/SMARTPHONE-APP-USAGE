import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)

file_path = r"mobile_usage_survey_dataset_5000.csv"
df = pd.read_csv(file_path)
# Print
print("Dataset Loaded Successfully")


df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)
# Print
print("\nColumns in Dataset:")
for col in df.columns:
    print(col)


def screen_time_to_hours(value):
    value = str(value).lower()
    if "2-4" in value:
        return 3
    elif "4-6" in value:
        return 5
    elif "6-8" in value:
        return 7
    elif "8" in value:
        return 9
    else:
        return np.nan

df["screen_time"] = df["average_screen_time_per_day"].apply(screen_time_to_hours)
df.dropna(subset=["screen_time"], inplace=True)


median_time = df["screen_time"].median()
df["usage_level"] = (df["screen_time"] > median_time).astype(int)

print("\nTarget Created (0 = Low Usage, 1 = High Usage)")


features = [
    "instagram",
    "whatsapp",
    "youtube",
    "snapchat",
    "time_spent_on_most_used_apps_hrs",
    "do_you_feel_you_use_your_phone_too_much?",
    "do_notifications_distract_you?",
    "phone_type"
]

le = LabelEncoder()
for col in features:
    df[col] = le.fit_transform(df[col].astype(str))

X = df[features]
y = df["usage_level"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight="balanced",
        random_state=42
    )
}

results = {}

print("\n------------- MODEL COMPARISON ------------------")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])


y_pred = best_model.predict(X_test)

print("\n----------- BEST MODEL PERFORMANCE -----------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure()
plt.hist(df["screen_time"], bins=5)
plt.xlabel("Screen Time (Hours)")
plt.ylabel("Number of Users")
plt.title("Distribution of Daily Screen Time")
plt.show()

plt.figure()
df["usage_level"].value_counts().plot(kind="bar")
plt.xlabel("Usage Level (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.title("High vs Low Smartphone Usage")
plt.show()

plt.figure()
plt.scatter(
    df["time_spent_on_most_used_apps_hrs"],
    df["screen_time"]
)
plt.xlabel("Time on Most Used Apps (hrs)")
plt.ylabel("Total Screen Time (hrs)")
plt.title("App Usage vs Total Screen Time")
plt.show()


ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

print("\nProject Execution Completed Successfully")
