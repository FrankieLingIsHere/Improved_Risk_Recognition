import pandas as pd
from risk_recognition.models.risk_classifier import RiskClassifier

# Load the combined improved CSV
csv_path = "risk_recognition/data/cleaned_accident_scenarios_combined_improved.csv"
df = pd.read_csv(csv_path)

# Rename columns to match expected names
# The classifier expects: scenario_text, risk_level, accident_cause
# Your CSV columns: Scenario, Risk_Level, Cause_of_Accident

df = df.rename(columns={
    "Scenario": "scenario_text",
    "Risk_Level": "risk_level",
    "Cause_of_Accident": "accident_cause"
})

# Drop rows with missing values in required columns
required_cols = ["scenario_text", "risk_level", "accident_cause"]
df = df.dropna(subset=required_cols)

# Initialize and train the classifier
classifier = RiskClassifier(base_model="all-MiniLM-L6-v2")
evaluation = classifier.train(
    df,
    test_size=0.2,
    fine_tune_epochs=3,
    classifier_epochs=10,
    learning_rate=0.001,
    batch_size=32
)

# Save the trained model
classifier.save_model("risk_recognition/models/trained_risk_cause_model")

# Print evaluation metrics
import pprint
pprint.pprint(evaluation)
