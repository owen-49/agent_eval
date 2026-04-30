import subprocess
import os
import json
import requests
from typing import Dict, Any

class ToolManager:
    def __init__(self):
        # 模拟 Token 存储，实际可从 .env 或数据库读取
        self._session_token = os.getenv("SESSION_TOKEN", "INIT_TOKEN_123")
        self.max_retries = 2

    def _refresh_token(self):
        """
        模拟动态刷新机制：运行时更新凭证
        在文书中可描述为“透明授权代理 (Transparent Authorization Proxy)”
        """
        print("[System] 检测到 Token 失效，正在触发动态刷新...")
        # 模拟 API 请求刷新
        # 正确写法：os 模块直接提供 urandom，没有 util 子模块
        self._session_token = "NEW_TOKEN_" + os.urandom(4).hex()
        return self._session_token

    def execute_python(self, code: str) -> str:
        """
        工具1：Python REPL (代码执行环境)
        硬性要求：通过 subprocess 实现基础隔离
        """
        try:
            # 限制执行时间为 5 秒，防止模型写出死循环导致系统卡死
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # 将标准错误回传，用于智能体的“自我纠错”
                return f"Execution Error:\n{result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (5s limit)."
        except Exception as e:
            return f"Unexpected Error: {str(e)}"

    def wikipedia_search(self, query: str) -> str:
        """
        工具2：信息检索工具
        包含 Token 校验逻辑，体现“运行时稳定性”
        """
        # 模拟 Token 校验失败的情况
        if "fail" in self._session_token.lower():
            return "Error: Unauthorized. Please refresh token."

        # 实际开发中可对接 MediaWiki API 或简单的 Search API
        # 这里为演示提供一个简化的逻辑
        return f"Search Result for '{query}': [Summary data from Wikipedia...]"

    def dispatch(self, tool_name: str, args: Any) -> str:
        """
        工具调度中枢：具备类型校验与自愈逻辑。
        旨在处理模型输出格式不稳定（格式坍缩）的情况。
        """
        # 1. 鲁棒性检查：如果 args 是字符串，尝试进行紧急解析
        if isinstance(args, str):
            try:
                # 尝试通过 json.loads 二次挽救
                import json
                args = json.loads(args)
            except:
                # 如果解析彻底失败，根据工具名强制包装成字典
                if tool_name == "python":
                    args = {"code": args}
                elif tool_name == "wikipedia":
                    args = {"query": args}
                else:
                    return f"Error: Tool '{tool_name}' parameters are in invalid format."

        # 2. 安全获取参数，防止 KeyError
        if tool_name == "python":
            code = args.get("code", "") if isinstance(args, dict) else args
            return self.execute_python(str(code))
        
        elif tool_name == "wikipedia":
            query = args.get("query", "") if isinstance(args, dict) else args
            observation = self.wikipedia_search(str(query))
            
            # 自愈逻辑：处理 Token 刷新中断
            if "Unauthorized" in observation:
                self._refresh_token()
                return "System Alert: Tool token refreshed. Please re-run your last action."
            
            return observation
        
        return f"Error: Tool '{tool_name}' not found."

# 快速测试脚本
if __name__ == "__main__":
    tm = ToolManager()
    # 测试代码执行
    print(tm.dispatch("python", {"code": "print(1+1)"}))
    # 测试超时安全
    print(tm.dispatch("python", {"code": "while True: pass"}))