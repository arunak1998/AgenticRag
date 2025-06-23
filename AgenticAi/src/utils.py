import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Any,Literal

from datetime import datetime
load_dotenv()

class CurrencyService:
    def __init__(self):
        self.api_key = os.getenv("CURRENCY_API_KEY")  # Optional, for future use if needed
        self.base_url = "https://api.exchangerate-api.com/v4/latest"

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies"""
        try:
            url = f"{self.base_url}/{from_currency}"
            response = requests.get(url)
            data = response.json()
            if response.status_code == 200 and to_currency in data.get('rates', {}):
                return data['rates'][to_currency]
            return 1.0
        except:
            return 1.0

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount from one currency to another"""
        rate = self.get_exchange_rate(from_currency, to_currency)
        return amount * rate


class TravelBudgetEstimator:
    def __init__(self, total_budget: float, num_days: int = 1, mode: str = "standard"):
        self.total_budget = total_budget
        self.num_days = max(1, num_days)
        self.mode = mode.lower()
        self.ratios = {
            "budget": {"stay": 0.35, "food": 0.30, "transport": 0.20, "activities": 0.15},
            "standard": {"stay": 0.40, "food": 0.30, "transport": 0.20, "activities": 0.10},
            "luxury": {"stay": 0.50, "food": 0.25, "transport": 0.15, "activities": 0.10}
        }

    def estimate_breakdown(self) -> dict:
        ratio = self.ratios.get(self.mode, self.ratios["standard"])
        return {k: round(self.total_budget * v, 2) for k, v in ratio.items()}

    def daily_budget(self) -> float:
        return round(self.total_budget / self.num_days, 2)
class TravelCostCalculator:
    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def calculate_total_cost(*costs: float) -> float:
        return sum(costs)

    @staticmethod
    def calculate_daily_budget(total_cost: float, days: int) -> float:
        return total_cost / days if days > 0 else 0


class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def get_current_weather(self, city: str) -> Dict:
        """Get current weather for a city"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": city,
                "appid": os.getenv("WEATHER_API"),
                "units": "metric"
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error in get_current_weather: {e}")
            return {}

    def get_weather_forecast(self, city: str, days: int = 5) -> Dict:
        """Get weather forecast for a city (3-hour interval data)"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": city,
                "appid": os.getenv("WEATHER_API"),
                "units": "metric",
                "cnt": days * 8
            }
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error in get_weather_forecast: {e}")
            return {}

class MarkdownExporter:
    def __init__(self, disclaimer: bool = True):
        self.include_disclaimer = disclaimer

    def export(self, response_text: str, filename: str = "ai_travel_plan.md") -> str:
        """
        Export travel plan to a well-formatted Markdown file.
        Includes metadata, disclaimer, and timestamp.
        """
        metadata = f"""# 🧳 AI Travel Itinerary

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Planner:** TravelAgent by Arun

---
"""
        disclaimer = (
            "\n---\n\n"
            "> ℹ️ *Note: This travel plan was AI-generated. Please verify all costs, places, and local regulations before booking.*"
            if self.include_disclaimer else ""
        )

        markdown_content = f"{metadata}\n{response_text}{disclaimer}"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"[Export] Markdown file saved: {filename}")
            return filename

        except Exception as e:
            print(f"[Export Error] Could not save file: {e}")
            return None