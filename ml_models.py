# ml_models.py
import os
import pickle   #saving ml models
import json     #reading/writing json files
from sklearn.ensemble import RandomForestRegressor #ML model to predict fatigue score
import numpy as np   #convert data into arrays for ml training

MODEL_PATH = "C:/Users/Sithumi/src/data/models/fatigue_model.pkl"
MODEL_META = MODEL_PATH + ".meta"          #how many rows the model was trained on
  
#predict fatigue score - using user data(if there are min 7 entries) or else using a dataset

def heuristic_fatigue_score(entry):
    steps = entry.get("steps", 0)
    sleep = entry.get("sleep", 0)
    water = entry.get("water", 0)
    mood = entry.get("mood", "Neutral").lower()
    score = 5.0
    if sleep < 6: score += 2.0
    elif sleep >= 8: score -= 1.0
    if water < 1.5: score += 1.0
    if steps < 3000: score += 0.5
    if mood in ["sad", "stressed", "tired"]: score += 1.5
    return max(0, min(10, score))

def _encode_mood(mood):
    # simple encoding for mood -> numeric
    mood = str(mood).lower()
    mapping = {
        "happy": 0,
        "okay": 1,
        "neutral": 1,
        "tired": 2,
        "stressed": 3,
        "sad": 3
    }
    return mapping.get(mood, 1)    #If the mood is not in the dictionary, it returns 1 (neutral/okay)

#train using user data
def train_fatigue_model_from_logs(log_path="C:/Users/Sithumi/src/data/logs.json", save_path=MODEL_PATH):
    if not os.path.exists(log_path):
        print(f"[ml_models] No logs found at {log_path}; skipping training.")
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:   #r-reading mode/utf-read all characters
            data = json.load(f)
    except Exception as e:
        print(f"[ml_models] Error reading logs {log_path}: {e}") #if anything wrong,print error e
        return None

    # gather records of user
    rows = []
    users = data.get("users", {})
    for uid, ud in users.items():
        for h in ud.get("history", []):
            # ensure required fields exist
            if all(k in h for k in ("steps", "sleep", "water", "mood")):
                rows.append({
                    "steps": float(h["steps"]),
                    "sleep": float(h["sleep"]),
                    "water": float(h["water"]),
                    "mood": h.get("mood", "Neutral")
                })

    if len(rows) < 7:
        # too few records for meaningful model
        print(f"[ml_models] Not enough records to train (need >=7, found {len(rows)})")
        return None

    X = []
    y = []
    for r in rows:
        X.append([r["steps"], r["sleep"], r["water"], _encode_mood(r["mood"])])
        y.append(heuristic_fatigue_score(r))

    X = np.array(X)
    y = np.array(y)

    #randomforest-learn patterns even from small dataset, use many small desicion trees,good at learning non linear patterns
    model = RandomForestRegressor(n_estimators=50, random_state=42) 
    model.fit(X, y)

    # save model + feature names
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:      #wb-write in binary
        pickle.dump((model, ["steps", "sleep", "water", "mood_encoded"]), f)

    # save the number of rows used to train
    meta = {"trained_on_rows": len(rows)}
    with open(MODEL_META, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[ml_models] Trained fatigue model on {len(rows)} records -> saved to {save_path}")
    return model, ["steps", "sleep", "water", "mood_encoded"]

def train_fatigue_model(csv_path="C:/Users/Sithumi/src/data/unified/train_data/train.csv", save_path=MODEL_PATH):
    """
    Legacy/training entrypoint: attempt to train from train.csv if available,
    else fall back to logs.json training.
    """
    # Try logs first (preferred for continuous learning)
    model_info = train_fatigue_model_from_logs(log_path="C:/Users/Sithumi/src/data/logs.json", save_path=save_path)
    if model_info:
        return model_info

    # If logs didn't suffice, use CSV 
    if os.path.exists(csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            # Expect columns-steps, sleep, water, mood
            if not set(["steps", "sleep", "water"]).issubset(df.columns):
                print("[ml_models] train.csv missing required columns; skipping CSV training.")
                return None
            X = []
            y = []
            for _, row in df.iterrows():
                mood = row.get("mood", "Neutral")
                X.append([row["steps"], row["sleep"], row["water"], _encode_mood(mood)])
                y.append(heuristic_fatigue_score({"steps":row["steps"], "sleep":row["sleep"], "water":row["water"], "mood":mood}))
            X = np.array(X)
            y = np.array(y)
            model = RandomForestRegressor(n_estimators=50, random_state=42)  #learn patterns even from very small data
            model.fit(X, y)
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump((model, ["steps", "sleep_hours", "water_intake", "mood"]), f)
            with open(MODEL_META, "w", encoding="utf-8") as f:
                json.dump({"trained_on_rows": len(df)}, f)
            print(f"[ml_models] Trained fatigue model from CSV -> saved to {save_path}")
            return model, ["steps", "sleep_hours", "water_intake", "mood"]
        except Exception as e:
            print(f"[ml_models] Error training from CSV: {e}")
            return None
    else:
        print("[ml_models] No CSV at path and no sufficient logs; no model trained.")
        return None

def load_fatigue_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                model, features = pickle.load(f)    #save model to the model path
            print(f"[ml_models] Loaded fatigue model from {path}")
            return model, features
        except Exception as e:
            print(f"[ml_models] Could not load model: {e}")
            return None, None
    return None, None
