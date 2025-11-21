"""
天氣 ReAct Agent（Nominatim + OpenWeather）

功能：
1. 當前天氣 (temperature, weather, humidity, wind)
2. 天氣預報 (5天)
3. 空氣質量 AQI (air quality)

改版重點：
- 不再使用 OpenWeather 的 geocoding API
- 新增 Nominatim 反查座標工具 get_coordinates（tool）
- 所有天氣相關工具改成吃 lat, lon，而不是地名字串
"""

import os
import time
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ==================== 常數設定 ====================

# Nominatim API（OpenStreetMap）
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT")


# ==================== OpenWeatherMap API ====================

class WeatherAPI:
    """OpenWeatherMap API 包裝（只負責用經緯度查資料）"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.weather_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        self.pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"

    # ==== 當前天氣 ====
    def get_current_weather_by_coords(self, lat: float, lon: float) -> dict:
        """用經緯度查當前天氣"""
        if not self.api_key:
            return {"error": "缺少 OPENWEATHER_API_KEY"}

        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "lang": "zh_cn",  # 中文描述
            }

            resp = requests.get(self.weather_url, params=params, timeout=5)
            if resp.status_code != 200:
                return {"error": f"OpenWeather API 錯誤：{resp.status_code}"}

            data = resp.json()
            main = data["main"]
            weather = data["weather"][0]
            wind = data.get("wind", {})

            location_name = data.get("name") or f"({lat:.3f}, {lon:.3f})"

            # 簡化的 UVI 計算（OpenWeather 免費版沒有直接 UVI）
            clouds = data.get("clouds", {}).get("all", 0)
            uvi = max(0, 10 - clouds / 10)

            return {
                "location": location_name,
                "temperature": round(main["temp"], 1),
                "feels_like": round(main["feels_like"], 1),
                "description": weather["description"],
                "humidity": main["humidity"],
                "pressure": main.get("pressure"),
                "wind_speed": round(wind.get("speed", 0.0), 1),
                "clouds": clouds,
                "uvi": round(float(uvi), 1),
                "lat": float(lat),
                "lon": float(lon),
            }

        except Exception as e:
            return {"error": f"當前天氣查詢失敗：{e}"}

    # ==== 預報 ====
    def get_forecast_by_coords(self, lat: float, lon: float, days: int = 5) -> dict:
        """用經緯度查 5 日天氣預報（每日彙整）"""
        if not self.api_key:
            return {"error": "缺少 OPENWEATHER_API_KEY"}

        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "lang": "zh_cn",
            }

            resp = requests.get(self.forecast_url, params=params, timeout=5)
            if resp.status_code != 200:
                return {"error": f"OpenWeather 預報 API 錯誤：{resp.status_code}"}

            data = resp.json()
            city = data.get("city", {})
            location_name = city.get("name") or f"({lat:.3f}, {lon:.3f})"

            from datetime import datetime

            forecasts_by_day = {}

            for item in data.get("list", []):
                dt = datetime.fromtimestamp(item["dt"])
                date = dt.strftime("%m-%d")

                if date not in forecasts_by_day:
                    forecasts_by_day[date] = {
                        "temps": [],
                        "descriptions": [],
                        "pop_values": [],
                    }

                forecasts_by_day[date]["temps"].append(item["main"]["temp"])
                forecasts_by_day[date]["descriptions"].append(
                    item["weather"][0]["description"]
                )
                forecasts_by_day[date]["pop_values"].append(item.get("pop", 0))

            forecasts = []
            for date in sorted(forecasts_by_day.keys())[:days]:
                f = forecasts_by_day[date]
                forecasts.append({
                    "date": date,
                    "temp_min": round(min(f["temps"]), 1),
                    "temp_max": round(max(f["temps"]), 1),
                    "description": f["descriptions"][0],
                    "precipitation_probability": round(max(f["pop_values"]) * 100, 1),
                })

            return {
                "location": location_name,
                "lat": float(lat),
                "lon": float(lon),
                "forecasts": forecasts,
            }

        except Exception as e:
            return {"error": f"預報查詢失敗：{e}"}

    # ==== 空氣品質 ====
    def get_air_quality_by_coords(self, lat: float, lon: float) -> dict:
        """用經緯度查空氣品質"""
        if not self.api_key:
            return {"error": "缺少 OPENWEATHER_API_KEY"}

        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
            }

            resp = requests.get(self.pollution_url, params=params, timeout=5)
            if resp.status_code != 200:
                return {"error": f"OpenWeather AQI API 錯誤：{resp.status_code}"}

            data = resp.json()
            if not data.get("list"):
                return {"error": "無空氣質量數據"}

            aqi_data = data["list"][0]
            main = aqi_data["main"]
            components = aqi_data["components"]

            aqi_value = main["aqi"]
            aqi_levels = {
                1: "優秀",
                2: "良好",
                3: "輕度污染",
                4: "中度污染",
                5: "重度污染",
            }

            return {
                "location": f"({lat:.3f}, {lon:.3f})",
                "lat": float(lat),
                "lon": float(lon),
                "aqi_value": aqi_value,
                "aqi_level": aqi_levels.get(aqi_value, "未知"),
                "pm25": round(components.get("pm2_5", 0), 1),
                "pm10": round(components.get("pm10", 0), 1),
                "o3": round(components.get("o3", 0), 1),
                "no2": round(components.get("no2", 0), 1),
                "so2": round(components.get("so2", 0), 1),
            }

        except Exception as e:
            return {"error": f"AQI 查詢失敗：{e}"}


# ==================== 初始化 API ====================

weather_api = WeatherAPI()


# ==================== 工具：地名 → 座標（Nominatim） ====================

@tool("get_coordinates")
def get_coordinates(location: str) -> str:
    """
    使用 Nominatim 依地名查詢可能的座標候選。

    Args:
        location: 任意地名（中文 / 英文皆可，例如「桃園」、「New York」）

    Returns:
        一段文字，列出最多 5 個候選地點及其座標，例如：
        1. Taiwan, Taoyuan... (lat: 24.993, lon: 121.301)
        2. China, Hunan, Taoyuan... (lat: ..., lon: ...)
    """
    try:
        params = {
            "q": location,
            "format": "json",
            "limit": 5,
            "addressdetails": 1,
        }
        headers = {
            "User-Agent": NOMINATIM_USER_AGENT,
        }

        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=8)
        if resp.status_code != 200:
            return f"Nominatim 錯誤：HTTP {resp.status_code}"

        data = resp.json()
        if not data:
            return f"找不到與「{location}」對應的地點，請嘗試提供更完整的地名（例如：城市 + 國家）。"

        lines = [f"以下是「{location}」對應的候選地點："]
        for idx, item in enumerate(data, start=1):
            display_name = item.get("display_name", "未知地名")
            lat = item.get("lat")
            lon = item.get("lon")
            country = item.get("address", {}).get("country", "")
            lines.append(
                f"{idx}. {display_name}（country: {country}, lat: {lat}, lon: {lon}）"
            )

        lines.append(
            "\n請根據需要選擇其中一個地點，並在後續工具呼叫中使用對應的 lat、lon 值。"
        )

        # Nominatim 有使用頻率限制，禮貌上稍微 sleep 一下（避免 Demo 連續轟炸）
        time.sleep(1)

        return "\n".join(lines)

    except Exception as e:
        return f"Nominatim 查詢失敗：{e}"


# ==================== 工具：用座標查天氣 / 預報 / AQI ====================

@tool("get_current_weather")
def get_current_weather(lat: float, lon: float) -> str:
    """
    用經緯度查詢當前天氣。

    Args:
        lat: 緯度（float）
        lon: 經度（float）

    Returns:
        一段描述當前天氣的文字。
    """
    result = weather_api.get_current_weather_by_coords(lat, lon)

    if "error" in result:
        return result["error"]

    return (
        f"{result['location']} 當前天氣：\n"
        f"- 溫度：{result['temperature']}°C（體感 {result['feels_like']}°C）\n"
        f"- 天氣：{result['description']}\n"
        f"- 濕度：{result['humidity']}%\n"
        f"- 風速：{result['wind_speed']} m/s\n"
        f"- 紫外線指數（估算）：{result['uvi']}\n"
        f"- 雲量：{result['clouds']}%"
    )


@tool("get_forecast")
def get_forecast(lat: float, lon: float) -> str:
    """
    用經緯度查詢 5 天天氣預報。

    Args:
        lat: 緯度（float）
        lon: 經度（float）

    Returns:
        整理後的 5 天預報資訊。
    """
    result = weather_api.get_forecast_by_coords(lat, lon, days=5)

    if "error" in result:
        return result["error"]

    output = [f"{result['location']} 5 天天氣預報："]
    for f in result["forecasts"]:
        output.append(
            f"\n日期：{f['date']}\n"
            f"  溫度：{f['temp_min']}~{f['temp_max']}°C\n"
            f"  天氣：{f['description']}\n"
            f"  降雨機率：{f['precipitation_probability']}%"
        )

    return "\n".join(output)


@tool("get_air_quality")
def get_air_quality(lat: float, lon: float) -> str:
    """
    用經緯度查詢空氣質量 AQI。

    Args:
        lat: 緯度（float）
        lon: 經度（float）

    Returns:
        包含 AQI、PM2.5 等資訊的文字。
    """
    result = weather_api.get_air_quality_by_coords(lat, lon)

    if "error" in result:
        return result["error"]

    return (
        f"{result['location']} 空氣質量：\n"
        f"- AQI：{result['aqi_value']}（{result['aqi_level']}）\n"
        f"- PM2.5：{result['pm25']} μg/m³\n"
        f"- PM10：{result['pm10']} μg/m³\n"
        f"- O3：{result['o3']} μg/m³\n"
        f"- NO2：{result['no2']} μg/m³\n"
        f"- SO2：{result['so2']} μg/m³"
    )


# ==================== Agent 系統 Prompt ====================

SYSTEM_PROMPT = """
你是一個專業的天氣預報員助手。

你有以下工具可以使用：
1. get_coordinates：用「地名」查詢可能的經緯度候選（使用 Nominatim）
2. get_current_weather：用「lat, lon」查詢當前天氣
3. get_forecast：用「lat, lon」查詢 5 天天氣預報
4. get_air_quality：用「lat, lon」查詢空氣質量 AQI

使用原則：

- 當使用者給的是地名（例如：「桃園天氣怎麼樣？」、「New York weather」），
  請先呼叫 get_coordinates 取得候選地點。
- 如果 get_coordinates 回傳多個候選，請「先向使用者確認」要哪一個，
  再使用對應的 lat、lon 呼叫其他工具。
- 當你已經知道明確的 lat、lon（可能是使用者直接給的，或上一輪已選定），
  就可以直接呼叫 get_current_weather / get_forecast / get_air_quality。

工具呼叫格式（由系統處理）：

- 對 get_coordinates：
  {"tool": "get_coordinates", "arguments": {"location": "<地名字串>"}}

- 對其他工具（舉例）：
  {"tool": "get_current_weather", "arguments": {"lat": 24.993, "lon": 121.301}}

請根據工具返回的信息，用友善和清晰的語言，用繁體中文回答使用者的問題。
"""


# ==================== ReAct Agent ====================

class WeatherReActAgent:
    """天氣 ReAct Agent（使用 LangGraph create_agent）"""

    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY")

        if not groq_api_key:
            raise ValueError("缺少 GROQ_API_KEY")
        if not openweather_api_key:
            raise ValueError("缺少 OPENWEATHER_API_KEY")

        # 初始化 LLM（需支援 tool calling）
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.3,
        )

        # 定義工具列表
        self.tools = [
            get_coordinates,
            get_current_weather,
            get_forecast,
            get_air_quality,
        ]

        # 記憶
        self.memory = InMemorySaver()

        # 建立 Agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
        )

        print("✅ 天氣 ReAct Agent（Nominatim + OpenWeather）已就緒\n")

    def query(self, question: str, thread_id: str = "default"):
        """查詢一次（程式內部用）"""
        print(f"👤 用戶：{question}\n")
        try:
            config = {"configurable": {"thread_id": thread_id}}

            resp = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )

            final_message = resp["messages"][-1].content
            print(f"🤖 助手：{final_message}\n")
            return final_message

        except Exception as e:
            print(f"❌ 錯誤：{e}\n")
            return f"發生錯誤：{e}"

    def chat(self):
        """命令列互動模式"""
        print("=" * 60)
        print("🌤️  天氣 ReAct Agent（Nominatim + OpenWeather）")
        print("=" * 60)
        print("\n功能：")
        print("  • 當前天氣 (溫度、濕度、風速、紫外線)")
        print("  • 5 天天氣預報 (包括降雨機率)")
        print("  • 空氣質量 AQI\n")
        print("示例：")
        print("  - 桃園天氣怎麼樣？")
        print("  - 東京未來 5 天的天氣預報")
        print("  - 香港的空氣質量如何？")
        print("  - exit 退出\n")

        thread_id = "main_conversation"

        while True:
            try:
                q = input("👤 你的問題：").strip()
                if q.lower() == "exit":
                    print("👋 再見！")
                    break
                if q:
                    self.query(q, thread_id=thread_id)
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break


def main():
    """主程式入口"""
    try:
        agent = WeatherReActAgent()
        agent.chat()
    except ValueError as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    main()
