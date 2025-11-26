# user_profile.py
import os
import json
import datetime
from continuous_learning import retrain_if_needed

class UserProfile:
    def __init__(self, user_id="user_1", log_path="C:/Users/Sithumi/src/data/logs.json"):
        self.user_id = user_id
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(log_path):
            # create initial structure
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"users": {self.user_id: {"history": [], "goals": {}}}}, f, indent=2)  #dump=save
        with open(log_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if "users" not in self.data:
            old_history = self.data.get("history", [])
            self.data = {"users": {self.user_id: {"history": old_history, "goals": {}}}}
            self._save()
        if self.user_id not in self.data["users"]:
            self.data["users"][self.user_id] = {"history": [], "goals": {}}
            self._save()
        self.default_goals = {"steps": 8000, "sleep": 7.5, "water": 2.0}  #if no goals are set for the user,system use these as default(at user's first time entry)

    def _save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

    def update_today(self, steps, sleep, water, mood):
        ts = datetime.datetime.utcnow().isoformat()   #utcnow-give current universal time,isoformat-makes it human-readable and storable in JSON.
        entry = {"timestamp": ts, "steps": int(steps), "sleep": float(sleep), "water": float(water), "mood": mood}
        self.data["users"][self.user_id]["history"].append(entry)
        self._save()

        # after saving, consider retraining the personal model
        try:
            retrain_if_needed(self)
        except Exception as e:
            print(f"[UserProfile] retrain_if_needed failed: {e}")

        return entry

    def get_history(self, days=None):
        hist = self.data["users"][self.user_id]["history"]
        if not days:
            return hist
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        filtered = []
        for h in hist:
            try:
                ts = datetime.datetime.fromisoformat(h["timestamp"])
                if ts >= cutoff:
                    filtered.append(h)
            except:
                continue
        return filtered

    def set_goal(self, key, value):      #Update a specific goal for the user
        self.data["users"][self.user_id]["goals"][key] = value
        self._save()

    def get_goals(self):                 #Return the current goals for the user
        user_goals = self.data["users"][self.user_id].get("goals", {})
        for k, v in self.default_goals.items():
            user_goals.setdefault(k, v)
        return user_goals

    def get_latest(self):                #Return the most recent entry in the user’s history
        hist = self.data["users"][self.user_id]["history"]
        return hist[-1] if hist else None

#UserProfile acts as the personal “memory” of the app, 
#storing and retrieving all user data for analysis, prediction, and personalized recommendations.