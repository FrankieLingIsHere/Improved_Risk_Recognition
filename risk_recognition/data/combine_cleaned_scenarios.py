import pandas as pd
import os

def combine_cleaned_files(paths, output_path):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        # Ensure Scenario is a readable sentence
        df['Scenario'] = df['Scenario'].apply(lambda x: str(x).strip().replace('  ', ' '))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    return combined

if __name__ == "__main__":
    cleaned_paths = [
        "risk_recognition/data/cleaned_face_mapped_final.csv",
        "risk_recognition/data/cleaned_osha_hse_final.csv",
        "risk_recognition/data/cleaned_osha_dataset_final.csv"
    ]
    output_path = "risk_recognition/data/cleaned_accident_scenarios_combined.csv"
    combine_cleaned_files(cleaned_paths, output_path)
    print(f"Combined cleaned file saved to {output_path}")
