import json
import re
import logging
from typing import List, Dict, Any
from core_function.tools import ToolManager
from core_function.parser import parse_agent_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReActAgent:
    def __init__(self, model_client: Any):
        self.client = model_client
        self.tools = ToolManager()
        self.max_steps = 10 
        self.turn_count = 0

    def _safe_extract(self, response_obj: Any) -> str:
        if isinstance(response_obj, str): return response_obj
        if hasattr(response_obj, 'choices'): return response_obj.choices[0].message.content
        if isinstance(response_obj, dict): return response_obj.get('content', str(response_obj))
        return str(response_obj)

    def run(self, question: str) -> str:
        self.system_prompt = """你是一个具备强鲁棒性的科研智能体。
[工具协议]
严格使用：<call name="工具名">{"参数": "值"}</call>
示例：<call name="wikipedia">{"query": "实体名"}</call>

[决策逻辑]
1. 每一轮必须先在 <thought> 中分析 Observation。
2. 如果 Observation 提示“精准页面未找到”，说明已进入模糊检索模式。你必须从碎片信息中提取线索，严禁在无新线索时重复搜索相同关键词。
3. 如果 Observation 包含答案线索，必须立即结束推理。

[输出硬约束]
1. 必须以 "Final Answer: [简洁答案]" 结尾。
2. 严禁在 Final Answer 标签后添加任何解释、句子或标点。"""

        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        self.turn_count = 0

        while self.turn_count < self.max_steps:
            self.turn_count += 1

            if self.turn_count > 1 and self.turn_count % 3 == 0:
                self.history.append({
                    "role": "system", 
                    "content": f"[指令重锚] 请保持 Final Answer 极其简洁，并优先从现有碎片信息中提取答案：\n{self.system_prompt}"
                })

            
            try:
                if hasattr(self.client, 'request_llm'):
                    raw_res = self.client.request_llm(self.history)
                else:
                    raw_res = self.client.generate(self.history)
                
                response = self._safe_extract(raw_res)
            except Exception as e:
                logging.error(f"LLM 调用失败: {e}")
                return f"Error: 模型连接中断"

            self.history.append({"role": "assistant", "content": response})

            if "<call" in response and not re.search(r'<call name="(.*?)">(.*?)</call>', response):
                feedback = "Error: 工具格式错误。请严格使用 <call name=\"...\">{\"...\": \"...\"}</call> 格式。"
                self.history.append({"role": "user", "content": feedback})
                continue

            thought, tool_name, tool_args = parse_agent_response(response)

            if "Final Answer:" in response:
                return response

            if tool_name:
                observation = self.tools.dispatch(tool_name, tool_args)
                
                if "碎片信息" in str(observation):
                    observation = f"{observation}\n[系统提示]：这已经是当前能获取的最全碎片信息。请基于此直接给出 Final Answer 或尝试搜索线索中提到的关联实体。"
                
                self.history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                self.history.append({"role": "user", "content": "请根据 Observation 给出 Final Answer 或使用 <call> 调用工具。"})

        return "Reached max steps without final answer."