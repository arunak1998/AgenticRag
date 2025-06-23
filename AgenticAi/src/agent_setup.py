from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.google_places.tool import GooglePlacesTool
from langchain_community.utilities.google_places_api import GooglePlacesAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from src.utils import (
    CurrencyService,
    TravelBudgetEstimator,
    TravelCostCalculator,
    WeatherService
)

from langchain_groq import ChatGroq
from typing import Dict, List, Any,Optional
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
from pydantic import BaseModel

class TripPlanInput(BaseModel):
    city: str
    origin: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class TotalExpenseInput(BaseModel):
    costs: List[float]
class AgentSetup:
    def __init__(self):
        # DuckDuckGo fallback
        self.search_tool = DuckDuckGoSearchRun()

        # Google Places
        try:
            google_places_key = os.getenv("GOOGLE_PLACES_API_KEY")
            if google_places_key:
                places_wrapper = GooglePlacesAPIWrapper(google_places_api_key=google_places_key)
                self.places_tool = GooglePlacesTool(api_wrapper=places_wrapper)
            else:
                self.places_tool = None
        except Exception:
            self.places_tool = None

        #  SerpAPI
        try:
            serpapi_key = os.getenv("SERPAPI_KEY")
            if serpapi_key:
                self.serp_search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
            else:
                self.serp_search = None
        except Exception:
            self.serp_search = None

        #  Serper
        try:
            serper_key = os.getenv("SERPER_API_KEY")
            if serper_key:
                self.serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_key)
            else:
                self.serper_search = None
        except Exception:
            self.serper_search = None

        # LLM Init
        self.llm = ChatOpenAI(model="gpt-4.1-2025-04-14")

        # Bind Tools
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.weather_service = WeatherService()
        self.currency_service = CurrencyService()

    def _setup_tools(self) -> List:
        """Define and bind all tools"""

        @tool
        def search_hotels(city: str,budget_range:str="Below-Expensive") -> str:
            """Search for hotels in a city using real-time tools like Google Places or search engines."""
            query = f"best {budget_range} accomedations hotels to stay in {city} with prices, reviews"

            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"hotels in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time hotel data: {places_result}"
                except Exception:
                    pass

            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Google hotels data: {serp_result}"
                except Exception:
                    pass

            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Latest hotels result: {serper_result}"
                except Exception:
                    pass

            return self.search_tool.invoke(query)

        @tool
        def search_flights(origin: str, destination: str, start_date: str, end_date: str) -> str:
            """
            Search for available round-trip flights between two cities within a date range.
            Args:
                origin: Departure city
                destination: Arrival city
                start_date: Trip start date (format: YYYY-MM-DD)
                end_date: Return date (format: YYYY-MM-DD)
            """
            query = (
                f"flights from {origin} to {destination} "
                f"departing on {start_date} and returning on {end_date} "
                f"with price, airline, and timing info"
            )

            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Flight options via SerpAPI: {serp_result}"
                except Exception:
                    pass

            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Flight info via Serper: {serper_result}"
                except Exception:
                    pass

            return self.search_tool.invoke(query)
        @tool
        def get_current_weather(city: str) -> str:
            """
        Get current weather conditions in a city.
            """
            data = self.weather_service.get_current_weather(city)

            if "main" in data:
                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                condition = data["weather"][0]["description"]
                wind = data["wind"]["speed"]

                print(  f"Current weather in {city}:\n"
                    f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                    f"Condition: {condition}\n"
                    f"Wind Speed: {wind} m/s")

                return (
                    f"Current weather in {city}:\n"
                    f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                    f"Condition: {condition}\n"
                    f"Wind Speed: {wind} m/s"
                )

            return "Weather data is currently unavailable. Try again later or check the city name."
        @tool
        def get_weather_forecast(city: str) -> str:
            """
            Get a 3-day weather forecast for a city.
            """
            forecast_data = self.weather_service.get_weather_forecast(city, days=3)

            if "list" in forecast_data:
                forecast_list = forecast_data["list"][:5]  # First few entries (next 15 hours)
                summary = [f"{f['dt_txt']}: {f['main']['temp']}°C, {f['weather'][0]['description']}"
                            for f in forecast_list]
                print(summary)
                return "\n".join(summary)

            return "Weather forecast unavailable. Please try another city or check spelling."

        @tool
        def search_restaurants(city: str) -> str:
            """Search for top-rated restaurants and food places in a city."""
            query = f"best restaurants and food places to eat in {city}"

            # Try Google Places
            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"restaurants in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time restaurant suggestions: {places_result}"
                except Exception:
                    pass

            # Try SerpAPI
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest from Google search: {serp_result}"
                except Exception:
                    pass

            # Fallback to DuckDuckGo
            return f"🍴 Suggestions via DuckDuckGo:\n{self.search_tool.invoke(query)}"

        @tool
        def search_transportation(city: str) -> str:
            """Search for public and private transportation options in a city."""
            query = f"transportation options in {city} including metro, bus, cab, Uber, and local transit"

            # SerpAPI
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Local transport options from Google search: {serp_result}"
                except Exception:
                    pass

            # Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Transit information from Serper: {serper_result}"
                except Exception:
                    pass

            # DuckDuckGo fallback
            return f"Transport info from DuckDuckGo:\n{self.search_tool.invoke(query)}"

        @tool
        def search_attractions(city: str) -> str:
            """Search for top tourist attractions and activities in a city using real-time data."""
            query = f"top attractions, activities, and things to do in {city}"

            # Google Places
            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"tourist attractions in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time attractions data: {places_result}"
                except Exception:
                    pass

            # SerpAPI
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest search results: {serp_result}"
                except Exception:
                    pass

            # Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Current search data: {serper_result}"
                except Exception:
                    pass

            # DuckDuckGo fallback
            return f"Attraction results from DuckDuckGo:\n{self.search_tool.invoke(query)}"

        @tool
        def estimate_trip_allocation(total_budget: float, num_days: int = 1, mode: str = "standard") -> str:
            """
            Estimate the budget allocation for a trip.

            Args:
                total_budget (float): Total available budget for the trip.
                num_days (int): Number of days for the trip.
                mode (str): Spending mode — can be "budget", "standard", or "luxury".

            Returns:
                str: A formatted string showing cost breakdown and daily budget.
            """
            estimator = TravelBudgetEstimator(total_budget, num_days, mode)
            breakdown = estimator.estimate_breakdown()
            daily = estimator.daily_budget()
            breakdown_str = "\n".join([f"{k.capitalize()}: ${v}" for k, v in breakdown.items()])
            return f"Estimated Budget Breakdown ({mode}):\n{breakdown_str}\n\nDaily Budget: ${daily}"


        @tool
        def estimate_hotel_cost(price_per_night: float, total_days: int) -> float:
            """
            Estimate the total cost of hotel stay.

            Args:
                price_per_night (float): Cost per night for accommodation.
                total_days (int): Total number of nights staying.

            Returns:
                float: Total hotel cost.
            """
            return TravelCostCalculator.multiply(price_per_night, total_days)


        @tool
        def add_costs(cost1: float, cost2: float) -> float:
            """
            Add two individual cost components together.

            Args:
                cost1 (float): First cost value.
                cost2 (float): Second cost value.

            Returns:
                float: Combined total cost.
            """
            return TravelCostCalculator.add(cost1, cost2)


        @tool(args_schema=TotalExpenseInput)
        def calculate_total_expense(costs: List[float]) -> float:
            """
            Sum multiple cost values to compute the total expense.

            Args:
                *costs (float): Any number of cost components.

            Returns:
                float: Total aggregated cost.
            """
            return TravelCostCalculator.calculate_total_cost(*costs)


        @tool
        def calculate_daily_budget(total_cost: float, days: int) -> float:
            """
            Calculate per-day budget from total cost and number of days.

            Args:
                total_cost (float): Overall expense.
                days (int): Total number of travel days.

            Returns:
                float: Daily cost estimate.
            """
            return TravelCostCalculator.calculate_daily_budget(total_cost, days)
        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
            """Convert currency between two types"""
            return self.currency_service.convert_currency(amount, from_currency, to_currency)

        @tool
        def get_exchange_rate(from_currency: str, to_currency: str) -> float:
            """Get live exchange rate between two currencies"""
            return self.currency_service.get_exchange_rate(from_currency, to_currency)


        @tool(args_schema=TripPlanInput)
        def create_trip_plan(city: str, origin: str, start_date: str = None, end_date: str = None) -> str:
            """
            Creates a full travel plan including:
            - Travel & Arrival Day (flights, check-in, optional evening)
            - Weather forecast
            - Hotel information
            - Dates & duration
            """

            try:
                today = datetime.today()

                # Handle start & end date
                if not start_date:
                    start = today + timedelta(days=1)
                    start_date = start.strftime("%Y-%m-%d")
                else:
                    start = datetime.strptime(start_date, "%Y-%m-%d")

                if not end_date:
                    end = start + timedelta(days=4)
                    end_date = end.strftime("%Y-%m-%d")
                else:
                    end = datetime.strptime(end_date, "%Y-%m-%d")

                # Validate range
                duration_days = (end - start).days + 1
                if duration_days <= 0:
                    return "Invalid date range. End date must be after start date."

                # Fetch data from tools (you must define these separately)
                flights = search_flights(origin, city, start_date, end_date)
                hotel_info = search_hotels(city)
                weather_forecast = get_weather_forecast(city, duration_days)

                # Extract one hotel for day 0 check-in
                hotel_line = hotel_info.split("\n")[0].strip() if hotel_info else "Hotel info not available"

                # Generate arrival day (Day 0)
                arrival_plan = f"""
        ## ✈️ Travel & Arrival – {start_date}

        - Depart from: {origin}
        - Arrive at: {city}
        - Suggested Hotel: {hotel_line}
        - Check-in and rest
        - Optional: Light walk nearby or dinner at a local spot

        ✈️ Flights:
        {flights}

        🏨 Hotel Info:
        {hotel_info[:300]}...
        """

                # Add weather forecast section
                weather_section = f"""
        🌦️ Weather Forecast for {city}:
        {weather_forecast[:300]}...
        """

                # Final formatted plan
                return f"""
        # 🌍 Trip Plan for {city}

        **Duration:** {duration_days} Days ({start_date} to {end_date})
        **From:** {origin}
        **To:** {city}

        ---

        {arrival_plan}

        ---

        {weather_section}
        """

            except Exception as e:
                return f"❌ Error creating trip plan: {str(e)}"

        @tool
        def create_day_plan(
                city: str,
                day_number: int,
                weather: str,
                attractions: str,
                restaurants: str = None,
                total_budget: float = 0,
                num_days: int = 1,
                mode: str = "standard"
            ) -> str:
                """
                Create a detailed day itinerary based on weather, attractions, food, and budget.
                """
                try:
                    # Extract top 2 attractions
                    top_attractions = [a.strip() for a in attractions.split(',') if a.strip()]
                    morning_spot = top_attractions[0] if len(top_attractions) > 0 else "local museum"
                    afternoon_spot = top_attractions[1] if len(top_attractions) > 1 else "historic walking trail"

                    # Restaurant formatting
                    top_restaurants = restaurants[:300].strip() + "..." if restaurants and len(restaurants) > 300 else restaurants

                    # Weather-aware planning
                    weather_lower = weather.lower()
                    if "rain" in weather_lower or "storm" in weather_lower or "showers" in weather_lower:
                        morning_plan = f"Visit {morning_spot} (indoor). Carry an umbrella!"
                        afternoon_plan = f"Indoor visit to {afternoon_spot} or explore a local café."
                        evening_plan = "Chill indoors with books, jazz bars, or a warm meal nearby."
                        tip = "☔ Rainy day — plan mostly indoor activities. Bring weather gear."
                    elif "cloud" in weather_lower:
                        morning_plan = f"Visit {morning_spot}. Keep flexible in case of drizzle."
                        afternoon_plan = f"Optional walk to {afternoon_spot} or relax in a café."
                        evening_plan = "Evening stroll or indoor performance nearby."
                        tip = "⛅ Cloudy skies — mix of indoor and light outdoor activities."
                    else:
                        morning_plan = f"Start with outdoor visit to {morning_spot}."
                        afternoon_plan = f"Continue to {afternoon_spot}, enjoy walking or biking."
                        evening_plan = "Enjoy city nightlife, rooftop dining, or river walk."
                        tip = "☀️ Sunny day — perfect for full outdoor sightseeing."

                    # 🧮 Budget breakdown
                    estimator = TravelBudgetEstimator(total_budget, num_days, mode)
                    breakdown = estimator.estimate_breakdown()
                    daily = estimator.daily_budget()
                    cost_block = "\n".join([f"- {k.capitalize()}: ${v}" for k, v in breakdown.items()])

                    return f"""
            🗓️ Day {day_number} in {city}
            🌤️ Weather: {weather}

            📍 Morning:
            - {morning_plan}

            📍 Afternoon:
            - {afternoon_plan}

            📍 Evening:
            - {evening_plan}
            - Dinner at: {top_restaurants if top_restaurants else 'a recommended local restaurant'}.

            💡 Tips: {tip}

            💰 Estimated Cost: ${daily}
            {cost_block}
            """.strip()

                except Exception as e:
                    return f"❌ Error generating day plan: {str(e)}"

        return [
    search_hotels,
    search_attractions,
    search_flights,
    search_restaurants,
    search_transportation,
    get_current_weather,
    get_weather_forecast,
    estimate_trip_allocation,
    estimate_hotel_cost,
    add_costs,
    calculate_total_expense,
    calculate_daily_budget,
    convert_currency,
    get_exchange_rate,
    create_trip_plan,
    create_day_plan,
]
