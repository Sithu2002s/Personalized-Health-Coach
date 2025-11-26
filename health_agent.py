# health_agent.py
import os
from rag_retriever import RAGRetriever
from recommender import recommend_goals

try:
    from gpt4all import GPT4All
except Exception:
    GPT4All = None

from ml_models import load_fatigue_model, heuristic_fatigue_score

class HealthCoachAgent:
    def __init__(self, user_profile, rag=None, llm_path="C:/Users/Sithumi/src/model/gpt4all/Phi-3-mini-4k-instruct.Q4_0.gguf"):
        self.user = user_profile      #This saves the user profile inside the agent
        self.rag = rag or RAGRetriever()      #search from the users
        self.llm = None
        if llm_path and GPT4All and os.path.exists(llm_path):  #check three conditions and then load if all ok
            try:
                self.llm = GPT4All(model_name=llm_path, device="cpu")
            except Exception:       #If anything goes wrong
                self.llm = None     #system uses fallback explanations instead of LLM text

        # Load the fatigue model 
        self.fatigue_model, self.fatigue_features = load_fatigue_model()

    def _query_llm(self, prompt, max_tokens=250):     #max length of the response
        if self.llm:
            try:
                resp = self.llm.generate(prompt, max_tokens=max_tokens)
                return resp.strip()   #remove extra spaces from the start/end of the response
            except Exception:         #if llm fails, do nothing
                pass
        return None

    def generate_advice(self, metrics):
        """High-level method to generate explainable advice."""
        query = f"best practices for steps {metrics['steps']}, sleep {metrics['sleep']}, water {metrics['water']}"
        resources = self.rag.retrieve(query, top_k=3)   #gets the best 3 documents that match the query.

        history = self.user.get_history(days=7)
        current_goals = self.user.get_goals()
        new_goals, reasons = recommend_goals(history, current_goals)   #recommend goals using rule base system(recommender)

        latest = metrics      #store incoming user data
        fatigue_score = None
        if self.fatigue_model:
            try:
                # create feature vector in the same order as training
                mood_enc = 0    #default value for the encoded mood.
                if "mood" in latest:
                    mood = latest.get("mood", "Neutral")
                    mapping = {"happy":0,"okay":1,"neutral":1,"tired":2,"stressed":3,"sad":3}  #Convert mood text → number
                    mood_enc = mapping.get(str(mood).lower(), 1)
                X = [[latest.get("steps",0), latest.get("sleep",0), latest.get("water",0), mood_enc]]
                fatigue_score = float(self.fatigue_model.predict(X)[0])   #float-converts the value to a normal number, 0-pick the first prediction from the list
            except Exception:
                fatigue_score = heuristic_fatigue_score(latest)
        else:
            fatigue_score = heuristic_fatigue_score(latest)

        bullets = []
        if metrics["steps"] < new_goals["steps"]:

            bullets.append(f"Walk: Try a 15–30 min walk (goal today: {new_goals['steps']} steps).")
        
        else:
            bullets.append("Nice! You've met today's step goal — keep the momentum.")

        if metrics["sleep"] < new_goals["sleep"]:
            bullets.append(f"Sleep: Aim to go to bed ~{int((new_goals['sleep']-metrics['sleep'])*60)} minutes earlier to reach {new_goals['sleep']}h.")
        else:

            bullets.append("Sleep: You're meeting your sleep goal. Maintain consistent bedtime.")

        if metrics["water"] < new_goals["water"]:
            bullets.append(f"Hydration: Increase intake toward {new_goals['water']} L. Small reminders every 1–2 hrs help.")
        else:

            bullets.append("Hydration: Good job keeping hydrated today.")

        if fatigue_score >= 7:
            bullets.append("Fatigue: Your recent signals show high fatigue — prioritize rest and avoid intense exercise today.")
        elif fatigue_score >= 4:
            bullets.append("Fatigue: Moderate fatigue detected — moderate activity and keep hydration up.")
        else:
            bullets.append("Energy: You're looking good — a moderate workout is fine if you feel up to it.")

        explanations = []
        
        explanations.extend(reasons[:2])  #came from the recommendation system

        if resources:
            source_snippets = [
                r["content"][:120].replace("\n", " ") + "..." #Take only the first 120 characters
                for r in resources
            ]
            explanations.append("Sources: " + "; ".join(source_snippets))

        prompt = "You are a concise friendly health coach. Convert these bullets into a friendly 4-6 short bullet advice with a 1-line explanation for each bullet.\n\nBullets:\n" + "\n".join(bullets) + "\n\nExplanations:\n" + "\n".join(explanations)
        llm_text = self._query_llm(prompt)
        if llm_text:
            final = llm_text
        else:
            final = "\n\n".join(["- " + b for b in bullets]) + "\n\nReasons:\n" + "\n".join(["- " + e for e in explanations])

        return {
            "advice_text": final,        #UI will show these
            "bullets": bullets,
            "reasons": explanations,
             }

    def proactive_actions(self):
        hist = self.user.get_history(days=3)
        actions = []
        if len(hist) >= 3:            #ensure we have 3days data

            avg_sleep = sum(h["sleep"] for h in hist)/len(hist)
            if avg_sleep < 6:
                actions.append("Detected <3 days low sleep → propose earlier bedtime and relaxation routine.")
                
        hist3 = self.user.get_history(days=3)
        if hist3:
            avg_water = sum(h["water"] for h in hist3)/len(hist3)
            if avg_water < 1.5:
                actions.append("Weekly hydration below recommended → suggest reminders and water-tracking.")
        return actions
