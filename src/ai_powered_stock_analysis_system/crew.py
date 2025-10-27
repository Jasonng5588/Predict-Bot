import os
import json
import random
from dotenv import load_dotenv

# Load .env
load_dotenv()

# =============================
# 🌐 Environment Config
# =============================
LITELLM_PROVIDER = os.getenv("LITELLM_PROVIDER", "custom")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1")
LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://127.0.0.1:11434/v1")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "dummy-key")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("LITELLM_API_BASE", "https://api.groq.com/openai/v1")


def get_random_groq_key():
    keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 5)]
    keys = [k for k in keys if k]
    return random.choice(keys) if keys else GROQ_API_KEY


# =============================
# 🧠 CrewAI Imports
# =============================
from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from src.ai_powered_stock_analysis_system.tools.yahoo_finance_data_tool import YahooFinanceDataTool
from src.ai_powered_stock_analysis_system.tools.lstm_stock_prediction_tool import LSTMStockPredictionTool
from src.ai_powered_stock_analysis_system.tools.xgboost_stock_prediction_tool import XGBoostStockPredictionTool
from src.ai_powered_stock_analysis_system.tools.daily_prediction_combiner_tool import DailyPredictionCombinerTool
from src.ai_powered_stock_analysis_system.tools.alpha_vantage_data_tool import AlphaVantageDataTool


# =============================
# 🧩 Helper: Choose Correct LLM
# =============================
def get_llm():
    """
    🧠 自动选择使用本地 Ollama 或 Groq（根据 .env 设置）
    """
    provider = (os.getenv("LITELLM_PROVIDER") or "custom").strip().lower()

    # ============ 🔹 本地 Ollama 配置 ============
    local_model = os.getenv("LOCAL_MODEL", "llama3.1")
    local_base = os.getenv("LOCAL_API_BASE", "http://127.0.0.1:11434")
    local_key = os.getenv("LOCAL_API_KEY", "dummy-key")

    # ============ 🔹 Groq 云端配置 ============
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_key = os.getenv("GROQ_API_KEY")
    groq_base = os.getenv("LITELLM_API_BASE", "https://api.groq.com/openai/v1")

    print(f"🔑 Provider: {provider}, API Key: {groq_key or local_key}")

    # ============ 🔍 根据 provider 自动选择 ============
    if provider == "custom":
        print(f"🔗 Using LOCAL Ollama model: {local_model}")
        return LLM(
            model=f"ollama/{local_model}",
            temperature=0.7,
            api_key=local_key,
            base_url=local_base
        )

    elif provider == "groq":
        print(f"☁️ Using Groq model: {groq_model}")
        return LLM(
            model=f"groq/{groq_model}",
            temperature=0.7,
            api_key=groq_key,
            base_url=groq_base
        )

    else:
        raise ValueError(f"❌ Unknown LITELLM_PROVIDER: {provider}")


# =============================
# 🤖 Crew Definition
# =============================
@CrewBase
class AiPoweredStockAnalysisSystemCrew:
    """AiPoweredStockAnalysisSystem crew"""

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            tools=[SerperDevTool(), YahooFinanceDataTool()],  # ✅ Added data tool
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=50,
            llm=get_llm(),
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],
            tools=[
                SerperDevTool(), 
                YahooFinanceDataTool(),  # ✅ CRITICAL: Added this!
                AlphaVantageDataTool()   # ✅ CRITICAL: Added this!
            ],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=50,
            llm=get_llm(),
        )

    @agent
    def investment_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["investment_advisor"],
            tools=[YahooFinanceDataTool()],  # ✅ Added data tool
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=50,
            llm=get_llm(),
        )

    @agent
    def ml_data_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_data_scientist"],
            tools=[
                YahooFinanceDataTool(),
                LSTMStockPredictionTool(),
                XGBoostStockPredictionTool(),
                DailyPredictionCombinerTool(),
                AlphaVantageDataTool()
            ],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=50,
            llm=get_llm(),
        )

    # =============================
    # 📋 Tasks
    # =============================
    @task
    def financial_research(self) -> Task:
        return Task(config=self.tasks_config["financial_research"], markdown=False)

    @task
    def technical_analysis(self) -> Task:
        return Task(config=self.tasks_config["technical_analysis"], markdown=False)

    @task
    def ml_prediction_analysis(self) -> Task:
        return Task(config=self.tasks_config["ml_prediction_analysis"], markdown=False)

    @task
    def short_term_price_prediction(self) -> Task:
        return Task(config=self.tasks_config["short_term_price_prediction"], markdown=False)

    @task
    def technical_analysis_summary(self) -> Task:
        return Task(config=self.tasks_config["technical_analysis_summary"], markdown=False)

    @task
    def investment_summary(self) -> Task:
        return Task(config=self.tasks_config["investment_summary"], markdown=False)

    @task
    def generate_menu_options(self) -> Task:
        return Task(config=self.tasks_config["generate_menu_options"], markdown=False)

    @crew
    def crew(self) -> Crew:
        """Creates the AiPoweredStockAnalysisSystem crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )