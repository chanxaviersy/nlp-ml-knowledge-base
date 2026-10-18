# 🤖 AI Agents 智能代理

> 摘要：AI Agent 是能夠自主規劃、調用工具、執行動作以達成目標的 LLM 系統。本章介紹 Agent 的核心架構、主流框架（LangChain、AutoGen、CrewAI）以及評估方法。  
> Abstract: AI Agents are LLM-based systems that autonomously plan, use tools, and execute actions to achieve goals. This chapter covers core agent architectures, mainstream frameworks, and evaluation methods.

---

## 1. Agent 的核心概念 | Core Concepts

### 1.1 什麼是 AI Agent？

**Agent = LLM + 記憶 + 規劃 + 工具使用**

```
┌────────────────────────────────────────────────────────────┐
│                       AI Agent                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LLM (大腦 / Brain)                       │  │
│  │  推理 / 決策 / 規劃                                  │  │
│  └────────────┬────────────────────────────┬────────────┘  │
│               │                            │                │
│  ┌────────────▼──────────┐  ┌─────────────▼────────────┐  │
│  │  記憶 (Memory)         │  │  工具 (Tools)              │  │
│  │  • 短期：對話上下文    │  │  • 搜尋引擎              │  │
│  │  • 長期：向量資料庫    │  │  • 代碼執行              │  │
│  │  • 情節：過去經驗      │  │  • API 調用              │  │
│  └────────────────────────┘  │  • 檔案操作              │  │
│                              └───────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          規劃 (Planning)                              │  │
│  │  • 任務分解                                            │  │
│  │  • ReAct / CoT / ToT                                 │  │
│  │  • 反思 / 自我修正                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Agent vs LLM

| 特性 | LLM | Agent |
|------|-----|-------|
| **輸入** | 用戶 query | 任務目標 |
| **輸出** | 文本回覆 | 動作序列 |
| **能力** | 純語言生成 | 語言 + 行動 |
| **互動** | 單輪 | 多輪、自主 |
| **工具** | 無 | 可調用 |
| **評估** | 文本質量 | 任務完成度 |

---

## 2. Agent 的核心循環 | Core Loop

### 2.1 ReAct 框架

**ReAct = Reasoning + Acting**，交錯進行推理和行動：

```
Thought 1: 我需要查詢今天的天氣
Action 1:  Search["今天台北天氣"]
Observation 1:  台北今天多雲，氣溫 25°C

Thought 2: 已知天氣，我可以推薦合適的活動
Action 2:  Finish["今天台北多雲 25°C，建議您去戶外散步"]
Observation 2: Task completed
```

```python
import re
from typing import List, Dict, Any

class ReActAgent:
    """ReAct Agent 簡化實現"""
    
    def __init__(self, llm, tools: Dict[str, callable], max_iterations=5):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
    
    def run(self, question: str) -> str:
        prompt = self._build_prompt(question)
        history = []
        
        for i in range(self.max_iterations):
            response = self.llm.invoke(prompt + "\n".join(history))
            
            # 解析 Thought / Action / Action Input
            thought_match = re.search(r"Thought: (.+)", response)
            action_match = re.search(r"Action: (.+)", response)
            input_match = re.search(r"Action Input: (.+)", response)
            
            if thought_match:
                history.append(f"Thought: {thought_match.group(1)}")
            
            # 如果是 Finish，結束
            if action_match and action_match.group(1).strip() == "Finish":
                if input_match:
                    return input_match.group(1).strip()
                return "Task completed"
            
            # 執行工具
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                
                if action in self.tools:
                    try:
                        observation = self.tools[action](action_input)
                    except Exception as e:
                        observation = f"Error: {e}"
                else:
                    observation = f"Unknown action: {action}"
                
                history.append(f"Action: {action}[{action_input}]")
                history.append(f"Observation: {observation}")
        
        return "Max iterations reached"
    
    def _build_prompt(self, question):
        tools_desc = "\n".join([f"- {name}: {fn.__doc__}" for name, fn in self.tools.items()])
        return f"""你是一個可以使用工具的 AI Agent。回答時使用以下格式：

Thought: 你的推理
Action: 工具名稱（Finish 表示結束）
Action Input: 工具輸入

可用工具：
{tools_desc}

問題：{question}
"""
```

### 2.2 Function Calling 模式

現代 LLM 通過 **Function Calling** 輸出結構化的工具調用：

```python
import json
from openai import OpenAI

client = OpenAI()

# 定義工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定城市的天氣",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名稱"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜尋網路資料",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜尋關鍵字"}
                },
                "required": ["query"]
            }
        }
    }
]

# 工具實現
def get_weather(city: str) -> str:
    # 實際調用天氣 API
    return f"{city} 今天晴，氣溫 25°C"

def search_web(query: str) -> str:
    return f"搜尋結果：關於 '{query}' 的相關資料..."

# Agent 流程
def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        
        msg = response.choices[0].message
        
        # 沒有工具調用，直接返回
        if not msg.tool_calls:
            return msg.content
        
        # 將 assistant 訊息加入歷史
        messages.append(msg)
        
        # 執行工具
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            
            if fn_name == "get_weather":
                result = get_weather(**fn_args)
            elif fn_name == "search_web":
                result = search_web(**fn_args)
            else:
                result = "Unknown tool"
            
            # 工具結果加入歷史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

print(run_agent("台北今天天氣如何？"))
```

---

## 3. 主流 Agent 框架 | Agent Frameworks

### 3.1 LangChain / LangGraph

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """在公司資料庫中搜尋資訊"""
    return f"找到 3 條關於 '{query}' 的記錄"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """發送電子郵件"""
    return f"郵件已發送至 {to}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_database, send_email]

# 創建 Agent
from langchain import hub
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 執行
result = agent_executor.invoke({
    "input": "幫我搜尋客戶訂單 #12345 的最新狀態，然後發郵件給客戶說明情況"
})
```

### 3.2 LangGraph：狀態機式 Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

def should_continue(state):
    messages = state["messages"]
    last = messages[-1]
    if "FINAL ANSWER" in last["content"]:
        return END
    return "tools"

# 構建狀態圖
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

### 3.3 AutoGen (Microsoft)

```python
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"model": "gpt-4o", "api_key": "..."}

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="你是一個數據分析助手。",
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # 人類只在終止時介入
    code_execution_config={"work_dir": "coding"},
    max_consecutive_auto_reply=10,
)

# 啟動對話
user_proxy.initiate_chat(
    assistant,
    message="幫我分析這份 CSV 數據，找出銷售趨勢",
)
```

### 3.4 CrewAI：多 Agent 協作

```python
from crewai import Agent, Task, Crew

# 定義 Agent
researcher = Agent(
    role='研究員',
    goal='找出 AI 最新發展',
    backstory='你是專注於 AI 領域的研究員',
    tools=[search_tool],
    llm=llm,
)

writer = Agent(
    role='作家',
    goal='撰寫引人入勝的 AI 文章',
    backstory='你是科技專欄作家',
    tools=[],
    llm=llm,
)

# 定義任務
research_task = Task(
    description='研究 2026 年 AI 領域的三大突破',
    agent=researcher,
    expected_output='詳細的研究報告',
)

writing_task = Task(
    description='基於研究結果撰寫 1500 字文章',
    agent=writer,
    expected_output='一篇完整的文章',
)

# 組建 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=2,
)

# 啟動
result = crew.kickoff()
```

---

## 4. Agent 規劃方法 | Planning Methods

| 方法 | 描述 | 適用場景 |
|------|------|----------|
| **CoT (Chain of Thought)** | 逐步推理 | 數學、邏輯題 |
| **ReAct** | 推理 + 行動交替 | 工具使用型任務 |
| **ToT (Tree of Thoughts)** | 樹狀搜索多個思路 | 需要探索的任務 |
| **Reflexion** | 自我反思 + 改進 | 需要迭代優化的任務 |
| **Plan-and-Execute** | 先規劃，後執行 | 結構化任務 |
| **AutoGPT** | 完全自主，目標驅動 | 開放性任務 |

### 4.1 Reflexion 自我反思

```python
class ReflexionAgent:
    """具有自我反思能力的 Agent"""
    
    def __init__(self, llm, max_trials=3):
        self.llm = llm
        self.max_trials = max_trials
        self.reflections = []
    
    def run(self, task):
        for trial in range(self.max_trials):
            # 1. 基於過去反思執行
            prompt = self._build_prompt(task, self.reflections)
            result = self.llm.invoke(prompt)
            
            # 2. 評估
            evaluation = self._evaluate(result)
            
            if evaluation["success"]:
                return result
            
            # 3. 反思失敗原因
            reflection = self.llm.invoke(
                f"任務：{task}\n"
                f"嘗試結果：{result}\n"
                f"失敗原因：{evaluation['reason']}\n"
                f"請反思如何改進："
            )
            self.reflections.append(reflection)
        
        return result
    
    def _evaluate(self, result):
        # 簡化：實際可用 LLM 評估或單元測試
        return {"success": "正確答案" in result, "reason": "..."}
```

---

## 5. 工具設計 | Tool Design

### 5.1 工具的組成

```python
@tool
def calculate(expression: str) -> str:
    """
    計算數學表達式
    
    Args:
        expression: 要計算的數學表達式，例如 "2 + 3 * 4"
    
    Returns:
        計算結果
    """
    try:
        result = eval(expression)
        return f"結果：{result}"
    except Exception as e:
        return f"計算錯誤：{e}"
```

**好工具的特徵**：
- ✅ 單一職責
- ✅ 清晰的 docstring
- ✅ 輸入輸出類型明確
- ✅ 錯誤處理完善
- ✅ 必要的參數校驗

### 5.2 常見工具類型

| 類型 | 例子 |
|------|------|
| **資訊檢索** | 搜尋引擎、向量資料庫、Wikipedia |
| **代碼執行** | Python 沙箱、SQL 執行 |
| **API 調用** | 天氣、新聞、股票、翻譯 |
| **檔案操作** | 讀寫 CSV/JSON/Excel |
| **通訊** | 發郵件、Slack、Teams |
| **知識查詢** | 計算器、單位轉換、字典 |
| **記憶** | 讀寫長期記憶 |

---

## 6. 記憶系統 | Memory Systems

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory
)

# 1. 緩衝記憶：保留所有對話
memory1 = ConversationBufferMemory()

# 2. 滑動窗口：只保留最近 k 輪
memory2 = ConversationBufferWindowMemory(k=5)

# 3. 摘要記憶：對歷史對話做摘要
memory3 = ConversationSummaryMemory(llm=llm)

# 4. 向量記憶：基於語義相似度檢索相關歷史
from langchain.vectorstores import FAISS
memory4 = VectorStoreRetrieverMemory(retriever=faiss_db.as_retriever(k=4))
```

---

## 7. Agent 評估 | Evaluation

### 7.1 評估維度

| 維度 | 衡量 |
|------|------|
| **任務完成率** | 是否達成最終目標 |
| **工具調用準確率** | 選對工具的比率 |
| **步驟效率** | 完成任務的平均步驟 |
| **錯誤恢復** | 失敗後自我修正的能力 |
| **幻覺率** | 編造事實的頻率 |
| **成本** | 達成任務的花費（API 調用次數、token） |

### 7.2 基準測試

| 基準 | 描述 |
|------|------|
| **AgentBench** | 多環境 Agent 評估 |
| **SWE-bench** | 軟件工程任務 |
| **WebArena** | 網頁瀏覽任務 |
| **GAIA** | 真實世界多步驟推理 |
| **ToolBench** | 工具使用能力 |

```python
# 使用 LangSmith 評估
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 定義評估函數
def correctness(run, example):
    score = 1.0 if run.outputs["output"] == example.outputs["expected"] else 0.0
    return {"score": score}

# 運行評估
results = evaluate(
    lambda x: agent_executor.invoke({"input": x["input"]}),
    data="agent-eval-dataset",
    evaluators=[correctness],
)
```

---

## 8. 安全與防護 | Safety

### 8.1 常見攻擊

| 攻擊 | 描述 |
|------|------|
| **Prompt Injection** | 注入惡意指令覆蓋系統提示 |
| **Tool Abuse** | 利用 Agent 調用危險工具 |
| **Data Exfiltration** | 通過工具洩露敏感數據 |
| **無限循環** | 設計不良的工具導致死循環 |
| **權限提升** | Agent 突破預定權限 |

### 8.2 防護措施

```python
# 1. 輸入過濾
def validate_input(user_input):
    if "忽略之前的指令" in user_input or "ignore previous" in user_input.lower():
        raise SecurityError("Potential prompt injection")
    return user_input

# 2. 工具白名單
ALLOWED_TOOLS = {"search", "calculate", "send_email"}

# 3. 操作確認
def execute_with_confirmation(tool_name, args):
    if tool_name in DANGEROUS_TOOLS:
        if not user_confirms(f"執行 {tool_name}? {args}"):
            raise PermissionError("User denied")
    return execute_tool(tool_name, args)

# 4. 輸出過濾
def filter_output(output):
    # 移除 PII（個人身份信息）
    output = remove_pii(output)
    # 檢測敏感內容
    if contains_harmful(output):
        return "[Content blocked]"
    return output

# 5. 速率限制
@rate_limit(calls=100, period="hour")
def agent_api():
    ...
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  Agent 設計模式選擇                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  簡單工具調用 → Function Calling (單次)                      │
│  多次工具調用 → ReAct / Tool-Use Agent                      │
│  複雜多步任務 → LangGraph 狀態機                              │
│  多角色協作 → AutoGen / CrewAI                              │
│  大規模生產 → 自定義框架 + 評估體系                          │
│                                                              │
│  關鍵原則：                                                  │
│  1. 工具設計要簡單、職責單一                                │
│  2. 始終設置最大迭代次數                                    │
│  3. 評估先行，建立基準                                      │
│  4. 安全防護貫穿全流程                                      │
│  5. 人類在迴路 (Human-in-the-Loop) 對關鍵操作確認            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
2. Wei, J., et al. (2022). Chain-of-Thought Prompting. *NeurIPS*.
3. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS*.
4. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.
5. Qin, Y., et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.

---

## ⚡ 5 分鐘快速實戰 | Quick Practice

### 1. LangChain 簡單 Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.llms import OpenAI

# 定義工具
def search_func(query: str) -> str:
    return f"Search result for: {query}"

tools = [
    Tool(name="Search", func=search_func, description="網路搜索"),
    Tool(name="Calculator", func=lambda x: eval(x), description="數學計算"),
]

# 創建 Agent
from langchain import hub
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=OpenAI(temperature=0), tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "What is 25 * 48?"})
```

### 2. Function Calling (OpenAI)

```python
from openai import OpenAI
import json

client = OpenAI()
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "獲取天氣",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Taipei?"}],
    tools=tools,
)
```

### 3. AutoGen 多 Agent

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    "assistant",
    llm_config={"model": "gpt-4"},
)
user_proxy = UserProxyAgent(
    "user",
    code_execution_config={"work_dir": "coding"},
)

user_proxy.initiate_chat(assistant, message="Plot a sine wave.")
```

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 構建 ReAct Agent：能調用「搜索」+「計算器」 |
| 2 | ⭐⭐ | 加入 Memory：對話歷史保留 |
| 3 | ⭐⭐ | Multi-Agent：研究員 + 寫作員 + 編輯 |
| 4 | ⭐⭐⭐ | 構建 Code Agent：自動寫 Python 並執行 |
| 5 | ⭐⭐⭐ | 構建 RAG Agent：先檢索再回答 |
| 6 | ⭐⭐⭐⭐ | 構建 Browser Agent：自動網頁操作 |

---

## 🎯 Agent 設計模式 | Design Patterns

| 模式 | 描述 | 適用 |
|------|------|------|
| **ReAct** | Reasoning + Acting | 通用 |
| **Reflexion** | 自我反思 | 錯誤恢復 |
| **Plan-and-Execute** | 先計劃再執行 | 複雜任務 |
| **Multi-Agent** | 多 Agent 協作 | 開放問題 |
| **Tool-Use** | 工具調用 | 結構化任務 |
| **Code-as-Action** | 代碼即動作 | 編程任務 |

---

*© 2026 chanxaviersy*
