
import json
import csv
import re
from textblob import TextBlob
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")

# Input/output paths
jsonl_path = "risk_recognition/data/situation_risk_predictions_mistral_standardized.jsonl"
output_csv = "risk_recognition/data/situation_risk_predictions_mistral_improved.csv"

def extract_details(scenario):
    pattern = r"(.*?) suffered (.*?) to the (.*?) while (.*?) with (.*?) in (.*?)."
    match = re.match(pattern, scenario)
    if match:
        role = match.group(1)
        injury = match.group(2)
        body_part = match.group(3)
        activity = match.group(4)
        hazard = match.group(5)
        context = match.group(6)
        return role, injury, body_part, activity, hazard, context
    return None, None, None, None, None, None

def improve_text(text):
    # Remove datetime info (e.g., 'It 1:30 p.m. on February 16, 2022,')
    text = re.sub(r"It [\d:apm\. ]+ on [A-Za-z]+ \d{1,2}, \d{4},?", "", text)
    # Remove other date/time patterns
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s?[ap]\.m\.)? on [A-Za-z]+ \d{1,2}, \d{4},?", "", text)
    text = re.sub(r"[A-Za-z]+ \d{1,2}, \d{4}", "", text)
    # Remove specific employee identifiers (Employee #1 -> Employee)
    text = re.sub(r"Employee\s*#\d+", "Employee", text)
    # Remove other similar patterns (Worker #2, Person #3)
    text = re.sub(r"(Worker|Person|Manager|Operator|Supervisor)\s*#\d+", r"\1", text)
    tb = TextBlob(text)
    corrected = str(tb.correct())
    readability = textstat.flesch_reading_ease(corrected)
    doc = nlp(corrected)
    entities = [ent.text for ent in doc.ents]
    # If scenario is too generic or repetitive, try to rewrite
    if "Lack of safety measures" in corrected:
        if entities:
            corrected = f"No protective gear or supervision for {', '.join(entities)}."
        else:
            corrected = corrected.replace("Lack of safety measures", "No protective gear or supervision")
    return corrected, readability

def make_specific_cause(role, injury, body_part, activity, hazard, context, scenario, orig_cause):
    # If original cause is generic or 'Other/Unspecified', build a more specific cause sentence
    if (
        "Lack of safety measures" in orig_cause
        or orig_cause.lower().startswith("seizure due to")
        or orig_cause.strip().lower() in ["other/unspecified", "other", "unspecified"]
    ):
        cause_parts = []
        if injury and hazard:
            cause_parts.append(f"{injury.capitalize()} caused by {hazard}")
        elif injury:
            cause_parts.append(f"{injury.capitalize()} (cause unspecified)")
        elif hazard:
            cause_parts.append(f"Incident caused by {hazard}")
        if activity:
            cause_parts.append(f"during {activity}")
        if body_part:
            cause_parts.append(f"affecting the {body_part}")
        if role:
            cause_parts.append(f"({role})")
        if context:
            cause_parts.append(f"in {context}")
        return ", ".join(cause_parts)
    # Otherwise, use original cause
    return orig_cause

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
            role, injury, body_part, activity, hazard, context = extract_details(improved_scenario)
            improved_cause = make_specific_cause(role, injury, body_part, activity, hazard, context, improved_scenario, cause)
            # Standardize risk level
            risk_map = {"High": "high", "Medium": "medium", "Low": "low"}
            risk_std = risk_map.get(risk, risk.lower())
            writer.writerow([improved_scenario, risk_std, improved_cause, readability])


def process_csv(input_csv, output_csv):
    with open(input_csv, "r", encoding="utf-8") as fin, open(output_csv, "w", newline='', encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        base_fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else ["Scenario", "Risk_Level", "Cause_of_Accident", "Keywords"]
        if "Readability_Score" not in base_fieldnames:
            fieldnames = base_fieldnames + ["Readability_Score"]
        else:
            fieldnames = base_fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            scenario = row["Scenario"]
            improved_scenario, readability = improve_text(scenario)
            role, injury, body_part, activity, hazard, context = extract_details(improved_scenario)
            orig_cause = row.get("Cause_of_Accident", "")
            improved_cause = make_specific_cause(role, injury, body_part, activity, hazard, context, improved_scenario, orig_cause)
            row["Scenario"] = improved_scenario
            row["Cause_of_Accident"] = improved_cause
            writer.writerow(row)

if __name__ == "__main__":
    # Uncomment one of the following to process either JSONL or CSV
    # process_jsonl(jsonl_path, output_csv)
    # print(f"Improved scenarios saved to {output_csv}")

    # For combined CSV
    input_csv = "risk_recognition/data/cleaned_accident_scenarios_combined.csv"
    output_csv = "risk_recognition/data/cleaned_accident_scenarios_combined_improved.csv"
    process_csv(input_csv, output_csv)
    print(f"Improved combined CSV saved to {output_csv}")
