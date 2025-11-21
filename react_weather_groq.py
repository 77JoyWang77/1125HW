"""
ReAct 天氣 Agent - Groq 免費版本
功能：使用 LangGraph 建立 ReAct Agent，查詢實時天氣信息
使用 Groq (免費 API)
"""

import os
import json
from typing import Annotated, Sequence
from datetime import datetime
from dotenv import load_dotenv
import requests

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# 加載環境變數
load_dotenv()

# ==================== 定義 Tools ====================

@tool
def get_current_weather(location: str) -> str:
    """
    獲取指定地點的當前天氣信息。
    
    Args:
        location: 地點名稱（例如：台北、東京、紐約）
    
    Returns:
        天氣信息的 JSON 字符串
    """
    try:
        # 使用開放 API (Open-Meteo，完全免費，無需密鑰)
        response = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json",
            timeout=5
        )
        
        if response.status_code == 200 and response.json().get("results"):
            geo_data = response.json()["results"][0]
            lat, lon = geo_data["latitude"], geo_data["longitude"]
            location_name = geo_data["name"]
            
            weather_response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature&timezone=auto",
                timeout=5
            )
            
            if weather_response.status_code == 200:
                weather = weather_response.json()["current"]
                return json.dumps({
                    "location": location_name,
                    "country": geo_data.get("country", ""),
                    "temperature": weather["temperature_2m"],
                    "feels_like": weather.get("apparent_temperature"),
                    "humidity": weather["relative_humidity_2m"],
                    "wind_speed": weather["wind_speed_10m"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "Open-Meteo API (免費)"
                }, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"API 調用失敗：{e}")
    
    # 備用模擬數據
    return json.dumps({
        "location": location,
        "temperature": 20,
        "humidity": 70,
        "wind_speed": 10,
        "note": "無法連接到天氣 API，返回默認值",
        "timestamp": datetime.now().isoformat()
    }, ensure_ascii=False)


@tool
def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    獲取指定地點的天氣預報。
    
    Args:
        location: 地點名稱
        days: 預報天數（默認 3 天，最多 7 天）
    
    Returns:
        天氣預報的 JSON 字符串
    """
    days = min(max(days, 1), 7)
    
    try:
        # 地理編碼
        response = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json",
            timeout=5
        )
        
        if response.status_code == 200 and response.json().get("results"):
            geo_data = response.json()["results"][0]
            lat, lon = geo_data["latitude"], geo_data["longitude"]
            location_name = geo_data["name"]
            
            # 獲取預報
            forecast_response = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max&timezone=auto&forecast_days={days}",
                timeout=5
            )
            
            if forecast_response.status_code == 200:
                data = forecast_response.json()
                daily = data["daily"]
                
                forecast_data = {
                    "location": location_name,
                    "forecast_days": days,
                    "days": []
                }
                
                for i in range(len(daily["time"])):
                    day = {
                        "date": daily["time"][i],
                        "max_temp": daily["temperature_2m_max"][i],
                        "min_temp": daily["temperature_2m_min"][i],
                        "precipitation": daily["precipitation_sum"][i],
                        "wind_speed": daily["wind_speed_10m_max"][i]
                    }
                    forecast_data["days"].append(day)
                
                return json.dumps(forecast_data, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"預報 API 調用失敗：{e}")
    
    # 備用
    return json.dumps({
        "location": location,
        "note": "無法獲取預報數據"
    }, ensure_ascii=False)


@tool
def compare_weather(location1: str, location2: str) -> str:
    """
    比較兩個地點的天氣。
    
    Args:
        location1: 第一個地點
        location2: 第二個地點
    
    Returns:
        兩個地點天氣的比較
    """
    try:
        weather1_str = get_current_weather(location1)
        weather2_str = get_current_weather(location2)
        
        weather1 = json.loads(weather1_str)
        weather2 = json.loads(weather2_str)
        
        comparison = {
            "location1": {
                "name": weather1.get("location"),
                "temperature": weather1.get("temperature"),
                "feels_like": weather1.get("feels_like"),
                "humidity": weather1.get("humidity")
            },
            "location2": {
                "name": weather2.get("location"),
                "temperature": weather2.get("temperature"),
                "feels_like": weather2.get("feels_like"),
                "humidity": weather2.get("humidity")
            },
            "comparison": {
                "temperature_difference": abs(
                    (weather1.get("temperature") or 0) - (weather2.get("temperature") or 0)
                ),
                "warmer_location": location1 if (weather1.get("temperature") or 0) > (weather2.get("temperature") or 0) else location2
            }
        }
        return json.dumps(comparison, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"比較失敗：{str(e)}"


# ==================== 定義 Agent State ====================

class AgentState(TypedDict):
    """Agent 狀態定義"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ==================== 定義 Agent Graph ====================

class WeatherAgent:
    """天氣 ReAct Agent - 使用 Groq 免費 API"""
    
    def __init__(self, groq_api_key=None):
        """
        初始化天氣 Agent
        
        Args:
            groq_api_key: Groq API 金鑰（免費）
        """
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("請設定 GROQ_API_KEY 環境變數或傳入參數")
        
        # 使用 Groq 的免費 API
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",  # 免費模型
            api_key=self.api_key,
            temperature=0.7,
            max_tokens=1000
        )
        
        self.tools = [get_current_weather, get_weather_forecast, compare_weather]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """建立 ReAct Agent 圖"""
        
        # 定義節點函數
        def call_model(state: AgentState):
            """調用模型"""
            system_message = """你是一個有幫助的天氣助手。
            你可以使用以下工具：
            1. get_current_weather - 獲取當前天氣
            2. get_weather_forecast - 獲取天氣預報
            3. compare_weather - 比較兩個地點的天氣

            當用戶詢問天氣時，使用適當的工具獲取信息，然後提供清晰的答案。
            如果用戶要求比較天氣，使用 compare_weather 工具。"""
            
            messages = [
                {"role": "system", "content": system_message},
                *state["messages"]
            ]
            
            response = self.model_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: AgentState):
            """判斷是否繼續執行"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # 如果有 tool_calls，則調用工具
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            
            # 否則結束
            return "end"
        
        # 建立 StateGraph
        workflow = StateGraph(AgentState)
        
        # 添加節點
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # 添加邊
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
        workflow.add_edge("tools", "agent")
        
        # 編譯圖
        return workflow.compile()
    
    def query(self, question: str) -> str:
        """
        提出問題
        
        Args:
            question: 問題內容
        
        Returns:
            答案
        """
        print(f"\n👤 用戶：{question}")
        
        try:
            # 調用圖
            result = self.graph.invoke({
                "messages": [HumanMessage(content=question)]
            })
            
            # 提取最後一條消息作為答案
            last_message = result["messages"][-1]
            answer = last_message.content if hasattr(last_message, "content") else str(last_message)
            
            print(f"🤖 助手：{answer}\n")
            return answer
        except Exception as e:
            error_msg = f"❌ 出錯：{str(e)}"
            print(error_msg)
            return error_msg
    
    def interactive_chat(self):
        """進行互動式對話"""
        print("\n" + "=" * 50)
        print("🤖 天氣 Agent 互動模式 (Groq 免費版)")
        print("=" * 50)
        print("你可以問的例子：")
        print("  - 台北現在天氣如何？")
        print("  - 東京的 3 天天氣預報")
        print("  - 紐約和倫敦哪裡更暖和？")
        print("  - 新加坡接下來 5 天的天氣")
        print("（輸入 'exit' 或 'quit' 退出）\n")
        
        while True:
            try:
                question = input("👤 你的問題：").strip()
                
                if question.lower() in ['exit', 'quit']:
                    print("👋 再見！")
                    break
                
                if not question:
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 發生錯誤：{e}\n")


def main():
    """主程式"""
    
    print("=" * 50)
    print("🚀 天氣 ReAct Agent 啟動 (Groq 免費版)")
    print("=" * 50)
    
    try:
        # 初始化 Agent
        agent = WeatherAgent()
    except ValueError as e:
        print(f"❌ 錯誤：{e}")
        print("\n請設定 GROQ_API_KEY：")
        print("  1. 訪問 https://console.groq.com")
        print("  2. 創建 API 金鑰")
        print("  3. 將金鑰添加到 .env 文件：GROQ_API_KEY=your-key")
        return
    
    # 測試一些範例
    print("\n" + "=" * 50)
    print("📝 測試範例")
    print("=" * 50)
    
    test_questions = [
        "台北現在天氣如何？",
        "紐約的 3 天天氣預報",
        "香港和新加坡哪裡天氣更好？"
    ]
    
    for q in test_questions:
        try:
            agent.query(q)
        except Exception as e:
            print(f"❌ 查詢失敗：{e}\n")
    
    # 進入互動模式
    agent.interactive_chat()


if __name__ == "__main__":
    main()