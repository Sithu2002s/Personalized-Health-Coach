# continuous_learning.py
import os    #Used for checking if files exist, reading file paths
import json   #Used for reading JSON files
from ml_models import train_fatigue_model_from_logs, MODEL_META , MODEL_PATH

#MODEL META - File path where training information is saved
#MODEL PATH - where the trained model will be saved

def _read_meta():
    if os.path.exists(MODEL_META):                   #if metadata file exists, open it
        try:                                            
            with open(MODEL_META, "r", encoding="utf-8") as f:
                return json.load(f)       #try to load json data
        except:
            return {}             #if file is broaken or unreadable, return an empty dictionary
    return {}                     #if file does not exist at all, return

def retrain_if_needed(user_profile, min_records=7, retrain_every=7):
    """
    Check user_profile log and retrain the fatigue model if:
      - model doesn't exist, and there are >= min_records
      - or number of records used increased by retrain_every
    This is intentionally conservative to avoid retraining on every write.
    """
    log_path = user_profile.log_path
    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # count usable entries
    count = 0             #Start with zero records.
    users = data.get("users", {})
    for uid, ud in users.items():
        for h in ud.get("history", []):
            if all(k in h for k in ("steps", "sleep", "water", "mood")):   #Check if this record contains all four required fields
                count += 1   #If all exist → this record is valid and we increase the count by 1.

    if count < min_records:
        # If less than 7 valid logs → model cannot be trained.
        return None

    meta = _read_meta() #reads a small JSON file that stores how many rows were used the last time the model was trained.
    trained_on = meta.get("trained_on_rows", 0)  #number of rows used previously.If it doesn't exist

    # Conditions to retrain:
    # count - current num of valid records
    # trained on - last trained entries count. if first time trained on is 0
    if trained_on == 0 or (count - trained_on) >= retrain_every:
        print(f"[continuous_learning] Retraining model: {trained_on} -> {count} records")
        model_info = train_fatigue_model_from_logs(log_path=log_path, save_path=MODEL_PATH)
        return model_info

    # no retrain needed
    return None

#For a new user, the model will train after the first 7 daily entries.
#Then it retrains every 7 additional new entries after that.