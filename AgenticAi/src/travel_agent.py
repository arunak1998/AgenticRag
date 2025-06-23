from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages import HumanMessage
MessagesState = dict[str, list[BaseMessage]]
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
class TravelAgent:
    def __init__(self, travel_planner):
        self.travel_planner = travel_planner
        self.system_prompt = self._build_system_prompt()
        self.graph = self._build_graph()

    def _build_system_prompt(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "You are an expert AI Travel Agent and Budget Planner.\n"
                "Your job is to plan trips to any city worldwide using real-time tools and data.\n\n"

                "🔒 RULES YOU MUST FOLLOW:\n"
                "1. ALWAYS use available tools to fetch live information. Do NOT guess.\n"
                "2. NEVER assume weather, hotel prices, or flight details — fetch them using tools.\n"
                "3. Do NOT delay replies. Return full and detailed response in ONE go.\n"
                "4. Respond only in clear, structured Markdown (with `##`, `-`, etc.).\n\n"

                "🛠️ TOOLS YOU MUST USE:\n"
                "- `search_hotels(city)`: Find real hotels (prefer those with views or near landmarks).\n"
                "- `search_flights(origin, city, start_date, end_date)`: Get real flight details.\n"
                "- `get_current_weather(city)`: For live weather now.\n"
                "- `get_weather_forecast(city)`: For the next few days of weather.\n"
                "- `search_attractions(city)`: Top local attractions.\n"
                "- `search_restaurants(city)`: Popular places to eat.\n"
                "- `search_transportation(city)`: Transit options (metro, taxis, etc).\n"
                "- `estimate_trip_allocation(budget)`: Distribute trip budget.\n"
                "- `estimate_hotel_cost(city, days)`: Approximate cost of stay.\n"
                "- `add_costs(...)`, `calculate_total_expense(...)`, `calculate_daily_budget(...)`: Handle cost math.\n"
                "- `convert_currency(amount, from, to)`: Convert between INR, USD, etc.\n"
                "- `create_trip_plan(city, origin, start_date, end_date)`: Plan flights + hotels + weather together.\n"
                "- `create_day_plan(...)`: For each day's activities, meals, weather, and cost.\n\n"


                "🧭 YOUR RESPONSE MUST INCLUDE:\n"
                "- ✅ A **day-by-day itinerary** (Day 1 to N) with:\n"
                "    • Morning, Afternoon, and Evening plans\n"
                "    • Attractions with names and short details\n"
                "    • Meals and restaurant suggestions (with type/cost if known)\n"
                "- ✈️ **Arrival and Return Flights** (including airline, cost, and timing if available)\n"
                "- 🏨 **Hotel Stay** info: name, location, and estimated price per night\n"
                "- 💰 **Budget breakdown**: Stay, Food, Transport, Activities\n"
                "- 🚕 **Transport info** (e.g. metro, local taxi, walking)\n"
                "- 🌦️ **Weather forecast** per day from tool \n\n"
                "📎 Assume nothing. **ALWAYS** use tools to fill in real data."
                "- Include **weather** (from tool) at the top of each day.\n"
                "- Add **a helpful travel tip** (based on weather, location, etc) at the end of the day section.\n"
                "💡 USE TOOLS Always:\n"
                "- The user provides destination or travel dates\n"
                "- You need current hotel, weather, or flight info\n"
                "- You need to estimate currency conversion or budget per day\n\n"

                "✍️ FORMAT:\n"
                "- Use Markdown: `##`, `-`, `**bold**`, bullet points\n"
                "- Each day: `### Day X: Title`, followed by Morning/Afternoon/Evening\n"
                "- Keep everything readable and informative\n"
                "- End with a **Notes** section if assumptions were made (e.g. flight prices vary)\n\n"

                "🚫 NEVER say: 'I'll look it up', 'Let me check', or 'Hold on'.\n"
                "Generate final answers confidently using tools if required."
            )
        )


    def _agent_node(self, state: MessagesState) -> MessagesState:
        """Invoke the LLM with system prompt + chat history."""
        messages = [self.system_prompt] + state["messages"]
        response = self.travel_planner.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        """Check whether to continue tool calls or finish."""
        last_message = state["messages"][-1]
        content = last_message.content.lower()

        if getattr(last_message, "tool_calls", None):
            return "tools"

        if any(p in content for p in [
            "let me search", "i'll look up", "please hold on",
            "i'll prepare", "let me gather", "i need to check"
        ]):
            return "tools"

        if len(content) < 500:
            return "tools"

        keywords = ["hotel", "attraction", "cost", "weather", "itinerary"]
        if sum(1 for k in keywords if k in content) < 3:
            return "tools"

        return "__end__"

    def _build_graph(self):
        """Builds and compiles the LangGraph agent workflow."""
        graph = StateGraph(MessagesState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.travel_planner.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._should_continue)
        graph.add_edge("tools", "agent")

        return graph.compile()

    def plan_trip(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Main entrypoint to plan a trip using LangGraph agent flow.

        - Executes graph with the user query.
        - Controls loop with max_iterations (recursion limit).
        - Triggers a summarization prompt if LLM response is short.
        - Falls back to a direct LLM response if toolchain fails.
        """
        messages = [HumanMessage(content=user_input)]
        config = {"recursion_limit": max_iterations}
        response_messages = messages  # fallback default
        final_response = ""

        try:
            # Stream the graph execution step-by-step
            stream = self.graph.stream({"messages": messages}, config=config)

            for step in stream:
                print("\n🔄 LangGraph Step Output")
                print("=" * 50)

                for key, value in step.items():
                    print(f"🧩 Key: {key}")
                    print("📤 Value:")

                    # If this is the main message stream
                    if key == "messages" and isinstance(value, list):
                        for msg in value:
                            try:
                                msg_type = getattr(msg, "type", "Unknown").capitalize()
                                content = getattr(msg, "content", "")
                                print(f"[{msg_type}] {content[:300]}...\n")
                            except Exception as e:
                                print(f"⚠️ Error printing message: {e}")
                        # Update the main message chain
                        response_messages = value
                    else:
                        print(value)

                    print("-" * 50)

            # Extract final response if any
            if response_messages:
                final_response = response_messages[-1].content

            # Fallback summarization if response seems short
            if len(final_response) < 800:
                summary_prompt = (
                    f"Generate a complete final summary based on the current planning data.\n\n"
                    f"Do NOT call any tools again. Just use the information you've already gathered.\n\n"
                    "### Respond in well-formatted Markdown, covering:\n"
                    "- ✅ Full day-by-day itinerary\n"
                    "- 📍 Top attractions\n"
                    "- 🍴 Restaurant suggestions\n"
                    "- 💸 Cost estimates\n"
                    "- 🌦️ Weather forecast summary\n"
                    "- 🚕 Transportation options\n\n"
                    f"Original user request: {user_input}"
                )
                summary_messages = response_messages + [HumanMessage(content=summary_prompt)]
                summary_response = self.travel_planner.llm_with_tools.invoke(summary_messages)
                return summary_response.content

            return final_response or "⚠️ No final response generated."

        except Exception as e:
            print(f"[TravelAgent] Graph execution error: {e}")
            return self._fallback_planning(user_input)

    def _fallback_planning(self, user_input: str) -> str:
        """
        Fallback if the graph fails or tool invocation breaks.
        Uses a direct LLM prompt to produce a complete travel plan without tools.
        """
        fallback_prompt = (
            f"Your task is to create a complete travel plan for the following user request:\n\n"
            f"🧳 **Trip Request**: {user_input}\n\n"
            "Build a comprehensive and realistic travel itinerary. Assume real-world knowledge, but do not use tools or real-time data.\n\n"
            "### Please include the following sections:\n"
            "1. **Day-by-Day Itinerary** – Include what to do each day, with times of day and key experiences.\n"
            "2. **Top Attractions** – Highlight must-see places and experiences.\n"
            "3. **Restaurant Suggestions** – Recommend food spots with cuisine types and approximate cost.\n"
            "4. **Estimated Budget** – Breakdown total cost into stay, food, activities, transport.\n"
            "5. **Weather Overview** – Describe the expected weather during travel (assume typical season).\n"
            "6. **Transportation Guide** – Explain how to get around the city (public, walking, ride apps).\n\n"
            "**Formatting instructions:**\n"
            "- Use clean Markdown with proper headers (##, ###, etc)\n"
            "- Use bullet points or lists where needed\n"
            "- Keep tone friendly but informative\n"
            "- DO NOT mention that tools were not used\n"
            "- DO NOT say 'I assume' or 'I think'\n"
        )

        messages = [self.system_prompt, HumanMessage(content=fallback_prompt)]
        response = self.travel_planner.llm_with_tools.invoke(messages)
        return response.content