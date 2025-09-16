import pandas as pd
import numpy as np
import re

# --- Utility functions ---
def map_risk_level(text):
    """
    Map injury severity or degree to standardized risk levels: high, medium, low
    """
    if pd.isna(text):
        return np.nan
    text = str(text).lower()
    # High risk indicators
    if any(word in text for word in ["fatal", "death", "amputation", "critical", "severe", "fracture", "hospitalized", "serious", "skull", "electrocute", "collapse", "crushed", "asphyxiated"]):
        return "high"
    # Medium risk indicators
    if any(word in text for word in ["moderate", "medium", "injury", "dislocation", "contusion", "bruise", "admitted", "broken", "burn", "concussion"]):
        return "medium"
    # Low risk indicators
    if any(word in text for word in ["minor", "nonfatal", "low", "minimal", "pain", "sprain", "fainted"]):
        return "low"
    return "medium"  # Default if unclear

def extract_keywords(text):
    """
    Extract keywords from a string (comma separated or space separated)
    """
    if pd.isna(text):
        return ""
    # Remove special chars except comma
    text = re.sub(r'[\|/;]', ',', str(text).lower())
    keywords = re.split(r",|\s", text)
    keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw) > 2]
    # Remove duplicates
    return ", ".join(sorted(set(keywords), key=keywords.index))

def clean_scenario_text(*args):
    """
    Combine multiple fields into a scenario text, removing NaNs and extra whitespace
    """
    parts = [str(a) for a in args if pd.notna(a) and str(a).strip()]
    # Remove duplicates and join
    return ". ".join(sorted(set(parts), key=parts.index)).strip()

def extract_cause(text, keywords=None):
    """
    Extract main cause from text or keywords
    """
    cause_types = [
        "fall", "electrocution", "struck by", "caught in", "chemical", "fire", "explosion", "collapse", "fatigue", "equipment failure", "human error", "process failure", "environmental", "asphyxiated", "burn", "crushed", "heart attack", "stroke", "seizure", "amputation", "fracture", "concussion"
    ]
    # Try keywords first
    if keywords:
        for kw in keywords.split(","):
            kw = kw.strip()
            for cause in cause_types:
                if cause in kw:
                    return cause
    # Then try text
    if pd.isna(text):
        return "other"
    text = str(text).lower()
    for cause in cause_types:
        if cause in text:
            return cause
    return "other"

# --- Cleaning functions for each dataset ---
def clean_face_mapped(path):
    df = pd.read_csv(path)
    # Scenario: use mapped fields and join with occupation, activity, and injury
    df["Scenario"] = df.apply(lambda row: clean_scenario_text(
        row.get("Occupation_Mapped"), row.get("Work Activity_Mapped"), row.get("Nature of Injury_Mapped"), row.get("Part of Body_Mapped"), row.get("Source of Injury_Mapped"), row.get("Industry Type_Mapped")), axis=1)
    # Risk_Level: map from Nature of Injury_Mapped and Part of Body
    df["Risk_Level"] = df.apply(lambda row: map_risk_level(str(row.get("Nature of Injury_Mapped", "")) + " " + str(row.get("Part of Body_Mapped", ""))), axis=1)
    # Cause_of_Accident: from mapped fields and scenario text
    df["Cause_of_Accident"] = df.apply(lambda row: extract_cause(
        clean_scenario_text(row.get("Source of Injury_Mapped", ""), row.get("Work Activity_Mapped", ""), row.get("Nature of Injury_Mapped", "")),
        clean_scenario_text(row.get("Source of Injury_Mapped", ""), row.get("Work Activity_Mapped", ""), row.get("Nature of Injury_Mapped", ""))), axis=1)
    # Keywords: combine mapped columns and remove duplicates
    df["Keywords"] = df.apply(lambda row: extract_keywords(
        clean_scenario_text(row.get("Occupation_Mapped"), row.get("Nature of Injury_Mapped"), row.get("Part of Body_Mapped"), row.get("Source of Injury_Mapped"), row.get("Work Activity_Mapped"), row.get("Industry Type_Mapped"))), axis=1)
    return df[["Scenario", "Risk_Level", "Cause_of_Accident", "Keywords"]]

def clean_osha_hse(path):
    df = pd.read_csv(path)
    # Scenario: use Abstract Text, fallback to Event Description
    df["Scenario"] = df["Abstract Text"].fillna(df["Event Description"])
    # Risk_Level: map from Degree of Injury, Nature of Injury, and Part of Body
    df["Risk_Level"] = df.apply(lambda row: map_risk_level(str(row.get("Degree of Injury", "")) + " " + str(row.get("Nature of Injury", "")) + " " + str(row.get("Part of Body", ""))), axis=1)
    # Cause_of_Accident: from Event Keywords, event_type, and scenario text
    df["Cause_of_Accident"] = df.apply(lambda row: extract_cause(
        clean_scenario_text(row.get("Event Keywords", ""), row.get("event_type", ""), row.get("Scenario", "")),
        row.get("Event Keywords", "")), axis=1)
    # Keywords: from Event Keywords and scenario text
    df["Keywords"] = df.apply(lambda row: extract_keywords(
        clean_scenario_text(row.get("Event Keywords", ""), row.get("Scenario", ""))), axis=1)
    return df[["Scenario", "Risk_Level", "Cause_of_Accident", "Keywords"]]

def clean_osha_dataset(path):
    df = pd.read_csv(path)
    # Scenario: use Abstract
    df["Scenario"] = df["Abstract"]
    # Risk_Level: map from Degree_of_Injury and Abstract
    df["Risk_Level"] = df.apply(lambda row: map_risk_level(str(row.get("Degree_of_Injury", "")) + " " + str(row.get("Abstract", ""))), axis=1)
    # Cause_of_Accident: from Keywords and Abstract
    df["Cause_of_Accident"] = df.apply(lambda row: extract_cause(
        clean_scenario_text(row.get("Keywords", ""), row.get("Abstract", "")),
        row.get("Keywords", "")), axis=1)
    # Keywords: from Keywords and Abstract
    df["Keywords"] = df.apply(lambda row: extract_keywords(
        clean_scenario_text(row.get("Keywords", ""), row.get("Abstract", ""))), axis=1)
    return df[["Scenario", "Risk_Level", "Cause_of_Accident", "Keywords"]]

# --- Main script ---
import json

def merge_csv_jsonl(csv_path, jsonl_path, scenario_col=None, output_path=None):
    df = pd.read_csv(csv_path)
    # Read JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        jsonl_data = [json.loads(line) for line in f]
    # Prepare output columns
    scenarios = []
    risk_levels = []
    causes = []
    keywords = []
    for i, row in enumerate(jsonl_data):
        output = row.get('output', {})
        # For face_mapped.csv, generate improved scenario and cause
        if 'face_mapped' in csv_path:
            if i < len(df):
                occ = df.iloc[i].get('Occupation_Mapped', '')
                inj = df.iloc[i].get('Nature of Injury_Mapped', '')
                body = df.iloc[i].get('Part of Body_Mapped', '')
                src = df.iloc[i].get('Source of Injury_Mapped', '')
                act = df.iloc[i].get('Work Activity_Mapped', '')
                ind = df.iloc[i].get('Industry Type_Mapped', '')
                # Scenario: natural sentence
                scenario = f"{occ} suffered {inj.lower()} to the {body.lower()} while {act.lower()} with {src.lower()} in {ind.lower()}.".replace('  ', ' ').strip().capitalize()
                # Cause: combine injury, source, activity
                cause = f"{inj} due to {src} during {act}".replace('  ', ' ').strip()
                # Keywords: combine mapped columns
                kw_list = [occ, inj, body, src, act, ind]
                keywords_str = ', '.join(sorted(set([k for k in kw_list if k and isinstance(k, str)]), key=kw_list.index))
            else:
                scenario = row.get('input', '')
                cause = output.get('Cause of Accident', '')
                keywords_str = ', '.join(output.get('Hazards', [])) if isinstance(output.get('Hazards', []), list) else str(output.get('Hazards', ''))
            scenarios.append(scenario)
            risk_levels.append(output.get('Degree of Injury', ''))
            causes.append(cause)
            keywords.append(keywords_str)
        else:
            # Scenario: from CSV if available, else from JSONL input
            if scenario_col and scenario_col in df.columns:
                scenario = str(df.iloc[i][scenario_col]) if i < len(df) else row.get('input', '')
            else:
                scenario = row.get('input', '')
            scenarios.append(scenario)
            risk_levels.append(output.get('Degree of Injury', ''))
            causes.append(output.get('Cause of Accident', ''))
            hazards = output.get('Hazards', [])
            keywords.append(', '.join(hazards) if isinstance(hazards, list) else str(hazards))
    # Build DataFrame
    merged_df = pd.DataFrame({
        'Scenario': scenarios,
        'Risk_Level': risk_levels,
        'Cause_of_Accident': causes,
        'Keywords': keywords
    })
    if output_path:
        merged_df.to_csv(output_path, index=False)
    return merged_df

if __name__ == "__main__":
    # Paths to your datasets and jsonl files
    merge_csv_jsonl(
        csv_path="risk_recognition/data/face_mapped.csv",
        jsonl_path="risk_recognition/data/situation_risk_predictions_mistral_standardized.jsonl",
        scenario_col=None, # Use JSONL input for scenario
        output_path="risk_recognition/data/cleaned_face_mapped_final.csv"
    )
    merge_csv_jsonl(
        csv_path="risk_recognition/data/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv",
        jsonl_path="risk_recognition/data/osha_model3_hf_format_cleaned_no_other.jsonl",
        scenario_col="Abstract Text",
        output_path="risk_recognition/data/cleaned_osha_hse_final.csv"
    )
    merge_csv_jsonl(
        csv_path="risk_recognition/data/osha_dataset.csv",
        jsonl_path="risk_recognition/data/construction_risk_natural_scenarios.jsonl",
        scenario_col="Abstract",
        output_path="risk_recognition/data/cleaned_osha_dataset_final.csv"
    )
    print("Merged CSVs with JSONL outputs. Cleaned files saved.")
