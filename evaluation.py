# evaluate_system.py
import time       #To measure how long certain operations take (like LLM response time).
from rag_retriever import RAGRetriever      #Retrieves information from a knowledge base for user queries.
from recommender import recommend_goals     #Adjusts user goals based on history (steps, sleep, water)
from health_agent import HealthCoachAgent   #Main AI agent for proactive health advice.
from user_profile import UserProfile        #Represents a user and stores their data.
import numpy as np                          #For calculating averages (mean), used for precision or timings.

def evaluate_rag():
    rag = RAGRetriever()
    test_queries = [
        "how to improve sleep",
        "how much water should I drink",
        "how to reduce stress",
        "how to increase daily steps",
        "tips for better hydration",
    ]

    precision_scores = []  #Lists to store precision and time taken for each query.
    retrieval_times = []

    for q in test_queries:             #for each test question
        start = time.time()            #start a timer
        results = rag.retrieve(q, top_k=3)  #retrieve top 3 docs
        end = time.time()                   #stop the timer
        retrieval_times.append(end - start) #save the time taken

        relevant = sum([1 for r in results if r.get("score",0) > 0.1])   #if results got score more than 0.1, then 1 will be added to the list
        precision_scores.append(relevant / max(1, min(3, len(results))))  #get the max from 1 and the other. consider 1,coz denominator cant be 0

#append - adds one value to the end of a list so you can collect multiple results

    return {
        "precision@3": np.mean(precision_scores),      #averages
        "avg_retrieval_time": np.mean(retrieval_times)
    }

def evaluate_goal_adjustment():
    history = [
        {"steps": 2000, "sleep": 5, "water": 1.0, "mood": "Sad"},
        {"steps": 2500, "sleep": 5.5, "water": 1.2, "mood": "Tired"},
        {"steps": 3000, "sleep": 6, "water": 1.3, "mood": "Okay"},
        {"steps": 3500, "sleep": 5.2, "water": 1.4, "mood": "Stressed"},
    ]                                                                      #Sample user history to test goal adjustment

    current_goals = {"steps": 8000, "sleep": 7.5, "water": 2.0}
    new_goals, reasons = recommend_goals(history, current_goals)

    return {
        "new_goals": new_goals,
        "reasons": reasons
    }

def evaluate_proactive_agent():
    user = UserProfile("test_user")
    agent = HealthCoachAgent(user)

    for i in range(3):
        user.update_today(steps=3000, sleep=5, water=1.2, mood="Tired")

    actions = agent.proactive_actions()
    return {
        "alerts_triggered": actions
    }

def evaluate_llm_response():
    user = UserProfile("test_user2")
    agent = HealthCoachAgent(user)

    metrics = {"steps": 4000, "sleep": 6, "water": 1.2, "mood": "Okay"}

    start = time.time()
    result = agent.generate_advice(metrics)
    end = time.time()

    return {
        "llm_response_time": end - start,
        "sample_advice": result["advice_text"]
    }

def run_all_evaluations():
    print("\n=== RAG Evaluation ===")
    print(evaluate_rag())

    print("\n=== Goal Adjustment Evaluation ===")
    print(evaluate_goal_adjustment())

    print("\n=== Proactive Agent Evaluation ===")
    print(evaluate_proactive_agent())

    print("\n=== LLM Performance Evaluation ===")
    print(evaluate_llm_response())

if __name__ == "__main__":
    run_all_evaluations()
