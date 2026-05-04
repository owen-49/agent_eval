import re
from core_function.client import AgentClient
from core_function.tools import ToolManager
from core_function.parser import parse_agent_response

class ReActAgent:
    def __init__(self, model_name="gpt-4"):
        self.client = AgentClient()
        self.tools = ToolManager()
        self.max_steps = 8  

    def run(self, question: str):
        system_prompt = """你是一个具备工具调用能力的智能体。请通过多步推理解决问题。
每次回复必须包含 <thought> 标签进行思考。
如果需要调用工具，请使用 <call name="工具名">{"参数": "值"}</call> 格式。
可选工具：
- wikipedia: 参数为 {"query": "搜索关键词"}
- python: 参数为 {"code": "python代码"}
当得到最终答案后，请输出：Final Answer: [你的答案]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        print(f"\n启动任务: {question}")

        for step in range(self.max_steps):
         
            response = self.client.request_llm(messages)
            if not response:
                return "Error: 模型响应中断"

            print(f"\n[Step {step+1}] Thought: \n{response}")

           
            thought, tool_name, tool_args = parse_agent_response(response)

            if "Final Answer:" in response:
                return response


            if tool_name:
                print(f"正在调用工具: {tool_name} | 参数: {tool_args}")
                observation = self.tools.dispatch(tool_name, tool_args)
                print(f"👁️  Observation: {observation}")

               
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "user", "content": "请继续推理并使用 <call> 调用工具，或给出 Final Answer。"})

        return "Reached max steps without final answer."