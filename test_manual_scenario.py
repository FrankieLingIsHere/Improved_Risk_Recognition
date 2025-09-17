import sys
from risk_recognition.models.risk_classifier import RiskClassifier

# Path to trained model directory
model_path = "risk_recognition/models/trained_risk_cause_model"

# Load trained model
classifier = RiskClassifier()
classifier.load_model(model_path)

print("Type a scenario below (or type 'exit' to quit):")
while True:
    scenario = input("Scenario: ").strip()
    if scenario.lower() == 'exit':
        print("Exiting.")
        break
    if not scenario:
        print("Please enter a non-empty scenario.")
        continue
    result = classifier.predict([scenario])
    print(f"Predicted Risk Level: {result['risk_levels'][0]}")
    print(f"Predicted Cause: {result['accident_causes'][0]}")
    print()
