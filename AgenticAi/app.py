# app.py

import streamlit as st
from datetime import datetime
from src.agent_setup import AgentSetup
from src.travel_agent import TravelAgent
from src.utils import MarkdownExporter

# -------------------- Setup --------------------
# Initialize LangGraph agent + planner
agent = AgentSetup()
planner = TravelAgent(agent)
exporter = MarkdownExporter()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("🌍 AI Travel Planner")
st.markdown("""
Plan your perfect trip with AI.
Just enter your travel request, and we'll generate a detailed plan including flights, hotels, food, and budget.
""")

# Default query
default_query = (
    "Plan a 6-day trip to London with hotel views, return flights, top food & historic places. "
    "Keep it within ₹1 lakh and convert everything to INR."
)

# User input
user_query = st.text_area("✏️ Describe your trip", value=default_query, height=150)

# Optional: Budget mode
mode = st.selectbox("Choose travel budget mode", ["budget", "standard", "luxury"])

# Button
if st.button("🧭 Generate Itinerary"):
    if not user_query.strip():
        st.warning("Please enter a trip description.")
    else:
        with st.spinner("Planning your trip..."):
            try:
                # Generate itinerary
                response = planner.plan_trip(user_query)

                st.subheader("✅ Your Travel Itinerary")
                st.markdown(response)

                # Export to Markdown
                filename = f"travel_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                saved_path = exporter.export(response, filename)

                # Download button
                with open(saved_path, "r", encoding="utf-8") as f:
                    st.download_button("📥 Download Markdown File", f, file_name=filename, mime="text/markdown")

            except Exception as e:
                st.error(f"❌ Oops, something went wrong:\n\n{e}")
