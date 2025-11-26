# recommender.py
from ml_models import heuristic_fatigue_score

def recommend_goals(history, current_goals):
    reasons = []
    if not history:
        return current_goals, ["No history: using default goals"]

    n = min(len(history), 7)            #looking last 7 days of user history and adjusts today’s goal
    last = history[-n:]
    avg_steps = sum(h["steps"] for h in last)/n
    avg_sleep = sum(h["sleep"] for h in last)/n
    avg_water = sum(h["water"] for h in last)/n

    new_goals = current_goals.copy()

    if avg_steps < current_goals["steps"] * 0.75: #if avg steps is less than the 0.75% from current goals, system add only 1000 steps additionally to the avg step to make achievable but never go below 3000 steps
        new_goals["steps"] = int(max(3000, avg_steps + 1000))
        reasons.append(f"Steps averaged {int(avg_steps)} → lowering target to {new_goals['steps']} to keep goals achievable.")
    elif avg_steps > current_goals["steps"] * 1.1:   #if avg step is higher, just praise and add only 500 additional steps to keep progress
        new_goals["steps"] = int(avg_steps + 500)
        reasons.append(f"Great consistency — raising steps target to {new_goals['steps']} to encourage progress.")

    if avg_sleep < current_goals["sleep"] - 0.75:
        new_goals["sleep"] = round(max(6.0, avg_sleep + 0.5),1)
        reasons.append(f"Average sleep {avg_sleep:.1f}h → set goal to {new_goals['sleep']}h to be realistic.")
    elif avg_sleep > current_goals["sleep"] + 0.5:
        new_goals["sleep"] = round(min(8.5, avg_sleep + 0.25),1)  # ensure max 8.5h
        reasons.append("You've been sleeping well — small increase to consolidate.")

    if avg_water < current_goals["water"]:
        new_goals["water"] = round(min(4.0, avg_water + 0.5),1)
        reasons.append(f"Hydration average {avg_water:.1f}L → suggestion: increase target to {new_goals['water']}L.")

    latest = last[-1] #checks the latest entry in the user’s history and estimates their fatigue score.
    fatigue = heuristic_fatigue_score(latest)
    if fatigue >= 7:
        reasons.append("High fatigue score detected → recommend light activity and rest.")
    elif fatigue <= 3:
        reasons.append("Low fatigue score → can try a moderate-intensity session.")
#4-6 fatigue score is normal. no special change

    return new_goals, reasons
