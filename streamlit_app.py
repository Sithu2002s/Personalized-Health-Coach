# app_streamlit.py
import streamlit as st
from user_profile import UserProfile
from health_agent import HealthCoachAgent
from rag_retriever import RAGRetriever
from utils import plot_history
from recommender import recommend_goals
import re

st.set_page_config(page_title="Personalized Digital Health Coach",
                   page_icon="💡", layout="wide")

# --------------------- UTILITY FUNCTION ---------------------
def summarize_text(text, max_chars=400):
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = ""
    for s in sentences:
        if len(summary) + len(s) <= max_chars:
            summary += s + " "
        else:
            break
    return summary.strip() + ("..." if len(summary) < len(text) else "")

# --------------------- PATHS & INIT ---------------------
RESOURCES_DIR = "C:/Users/Sithumi/src/data/resources"
LOG_PATH = "C:/Users/Sithumi/src/data/logs.json"

user = UserProfile(user_id="sithumi", log_path=LOG_PATH)
rag = RAGRetriever(resources_path=RESOURCES_DIR)
agent = HealthCoachAgent(user, rag=rag)

st.title("🤖 Personalized Digital Health Coach")

# --------------------- TABS ---------------------
tab1, tab2 = st.tabs(["Assistant", "Visualizations"])

# --------------------- TAB 1: ASSISTANT ---------------------
with tab1:
    with st.expander("Update Last Metrics", expanded=True):  #expanded-hidden(collapsed) when app loads
        col1, col2, col3, col4 = st.columns(4)
        steps = col1.number_input("Steps", min_value=0, value=5000, step=100)
        sleep = col2.number_input("Sleep (hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.25)
        water = col3.number_input("Water (L)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        mood = col4.selectbox("Mood", ["Happy", "Okay", "Sad", "Stressed", "Tired"])

        if st.button("Update & Get Advice"):

            entry = user.update_today(steps, sleep, water, mood)
            st.success("Metrics saved.")

            # ---------------- Personalized Advice ----------------
            adv = agent.generate_advice(entry)
            with st.expander("🧠 Personalized Advice", expanded=True):
                st.markdown(adv["advice_text"].replace("\n", "  \n"))

            with st.expander("Why this advice"):
                for r in adv["reasons"]:
                    st.markdown(f"- {r}")

            # ---------------- Today Goal ----------------
            history = user.get_history(days=7)
            current_goals = user.get_goals()

            # Using recommend_goals logic to show today's numeric targets
            new_goals, goal_reasons = recommend_goals(history, current_goals)

            with st.expander("🎯 Today's Goals", expanded=True):
                st.markdown(f"- Steps goal: **{new_goals['steps']}** steps")
                st.markdown(f"- Sleep goal: **{new_goals['sleep']}** hours")
                st.markdown(f"- Water goal: **{new_goals['water']}** L")
                

    # ---------------- Proactive Actions ----------------
    st.subheader("🔔 Proactive Actions")

    actions = agent.proactive_actions()
   
   
   
   
    if actions:
        for a in actions:
            st.warning(a)
    else:
        st.write("No proactive alerts at this moment.")

    # ---------------- Search Trusted Resources ----------------
    st.subheader("🔎 Search Trusted Resources")
    q = st.text_input("Any health related questions ?")

    if q:
        results = rag.retrieve(q, top_k=3)
        if results:
            combined_text = "\n\n".join([r['content'] for r in results])
            prompt = f"""
You are a friendly and practical health assistant. 
Use the information from the resources below to answer the user's question.
Do NOT repeat the resources verbatim. Give concise, actionable, and easy-to-understand advice.

User question: {q}

Resources:
{combined_text}

Answer:
"""
            answer = agent._query_llm(prompt)
            if answer:
                st.markdown(answer.replace("\n", "  \n"))
            else:
                st.info("LLM could not generate an answer at this moment.")
        else:
            st.info("No relevant resources found.")

# --------------------- TAB 2: VISUALIZATIONS ---------------------
with tab2:
    st.subheader("📊 History & Trends")
    history = user.get_history()

    if history:
        fig_steps, fig_sleep, fig_water = plot_history(history[-21:])
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### 🏃 Steps Trend")
            st.pyplot(fig_steps, use_container_width=True) #suse_container_width-tretches the content to fill the space

        with col2:
            st.write("### 😴 Sleep Trend")
            st.pyplot(fig_sleep, use_container_width=True)

        with col3:
            st.write("### 💧 Water Intake Trend")
            st.pyplot(fig_water, use_container_width=True)

        st.markdown("### **Last 5 Records**")
        for h in history[-5:][::-1]:
            st.markdown(
                f"- {h['timestamp'][:19]} — "
                f"**Steps:** {h['steps']}, "
                f"**Sleep:** {h['sleep']}h, "
                f"**Water:** {h['water']}L, "
                f"**Mood:** {h['mood']}"
            )
    else:
        st.info("No history yet. Add entries to begin personalization.")

# --------------------- FOOTER ---------------------
st.write("---")
st.markdown("""
**Privacy notice:** This health coach prototype collects and stores your personal activity data (steps, sleep, water intake, mood) locally in data/logs.json. 
- Your data is kept private and is not shared externally.
- You can delete your data anytime by removing the logs.json file.
""")
