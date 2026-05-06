import json
import re
import logging
from typing import List, Dict, Any
from core_function.tools import ToolManager
from core_function.parser import parse_agent_response

class ReActAgent:
    def __init__(self, model_client: Any):
        self.client = model_client
        self.tools = ToolManager()
        self.max_steps = 10 
        self.turn_count = 0

    def _extract_content(self, raw_response: Any) -> str:
        if isinstance(raw_response, str): return raw_response
        if hasattr(raw_response, 'choices'): return raw_response.choices[0].message.content
        if isinstance(raw_response, dict): return raw_response.get('content', str(raw_response))
        return str(raw_response)

    def run(self, question: str) -> str:
        # 1. 强化版 System Prompt：增加 Few-shot 示例防止格式偏移
        self.system_prompt = """你是一个具备自主自愈能力的科研智能体。
[工具格式]
必须严格使用：<call name="工具名">{"参数": "值"}</call>
示例：<call name="wikipedia">{"query": "OpenAI"}</call>

[行为准则]
1. 遇到工具返回的 'Error' 或 '线索提示'，必须在 <thought> 中分析原因并调整搜索词。
2. Final Answer 必须极其简洁，严禁任何解释。"""

        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        self.turn_count = 0

        while self.turn_count < self.max_steps:
            self.turn_count += 1
            
            
            if self.turn_count % 3 == 0:
                self.history.append({"role": "system", "content": f"[指令刷新]\n{self.system_prompt}"})

            
            try:
                raw_res = self.client.request_llm(self.history) 
                response = self._extract_content(raw_res)
            except AttributeError:
                
                raw_res = self.client.generate(self.history)
                response = self._extract_content(raw_res)
            
            self.history.append({"role": "assistant", "content": response})

            
            if "<call" in response and not re.search(r'<call name="(.*?)">(.*?)</call>', response):
                feedback = "Error: 检测到无效的工具格式。请严格使用 <call name=\"...\">{\"...\": \"...\"}</call> 且确保 JSON 双引号闭合。"
                self.history.append({"role": "user", "content": feedback})
                continue

            
            thought, tool_name, tool_args = parse_agent_response(response)

            if "Final Answer:" in response:
                return response

            if tool_name:
               
                observation = self.tools.dispatch(tool_name, tool_args)
                
                
                self.history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                
                self.history.append({"role": "user", "content": "请根据 Observation 继续推理，或给出 Final Answer。"})

        return "Reached max steps without final answer."