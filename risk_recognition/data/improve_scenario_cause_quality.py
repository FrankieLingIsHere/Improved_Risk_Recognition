import json
import csv
from textblob import TextBlob
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")

# Input/output paths
jsonl_path = "risk_recognition/data/situation_risk_predictions_mistral_standardized.jsonl"
output_csv = "risk_recognition/data/situation_risk_predictions_mistral_improved.csv"

def improve_text(text):
    # Grammar and spelling correction
    tb = TextBlob(text)
    corrected = str(tb.correct())
    # Readability score
    readability = textstat.flesch_reading_ease(corrected)
    # Entity extraction
    doc = nlp(corrected)
    entities = [ent.text for ent in doc.ents]
    # If scenario is too generic or repetitive, try to rewrite
    if "Lack of safety measures" in corrected:
        # Try to use entities or hazards for more specificity
        if entities:
            corrected = f"No protective gear or supervision for {', '.join(entities)}."
        else:
            corrected = corrected.replace("Lack of safety measures", "No protective gear or supervision")
    return corrected, readability

def improve_cause(cause, scenario):
    # If cause is generic, rewrite using scenario context
    if "Lack of safety measures" in cause:
        doc = nlp(scenario)
        hazards = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "EVENT", "ORG"]]
        if hazards:
            return f"No protective gear or supervision for {', '.join(hazards)}."
        else:
            return cause.replace("Lack of safety measures", "No protective gear or supervision")
    return cause

def process_jsonl(jsonl_path, output_csv):
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(output_csv, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Scenario", "Risk_Level", "Cause_of_Accident", "Readability_Score"])
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            scenario = obj.get("input", "")
            output = obj.get("output", {})
            cause = output.get("Cause of Accident", "")
            risk = output.get("Degree of Injury", "")
            # Improve scenario and cause
            improved_scenario, readability = improve_text(scenario)
            improved_cause = improve_cause(cause, improved_scenario)
            # Standardize risk level
            risk_map = {"High": "high", "Medium": "medium", "Low": "low"}
            risk_std = risk_map.get(risk, risk.lower())
            writer.writerow([improved_scenario, risk_std, improved_cause, readability])

if __name__ == "__main__":
    process_jsonl(jsonl_path, output_csv)
    print(f"Improved scenarios saved to {output_csv}")
