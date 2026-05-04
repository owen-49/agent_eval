import re
import json

def parse_agent_response(text):
    # 提取思维过程
    thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""


    call_match = re.search(r'<call name="(.*?)">(.*?)</call>', text, re.DOTALL)
    
    if call_match:
        tool_name = call_match.group(1).strip()
        tool_args_str = call_match.group(2).strip()
        try:
            # 尝试将参数解析为 JSON
            tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError:
            tool_args = tool_args_str 
        
        return thought, tool_name, tool_args
    
    return thought, None, None