import subprocess
import os
import wikipedia
import json
import re
from typing import Dict, Any, List, Tuple, Set

class ToolManager:
    def __init__(self):
        # 凭证管理
        self._session_token = os.getenv("SESSION_TOKEN", "INIT_TOKEN_123")
        
        # --- 核心：分层记忆架构 ---
        self.raw_buffer: List[str] = []
        self.entity_library: Set[str] = set()
        self.max_buffer_size = 3

    def _refresh_token(self):
        print("[System] 检测到 Token 失效，正在触发动态自愈刷新...")
        self._session_token = "NEW_TOKEN_" + os.urandom(4).hex()
        return self._session_token

    def _update_memory(self, new_observation: str):
        self.raw_buffer.append(new_observation)
        if len(self.raw_buffer) > self.max_buffer_size:
            
            old_obs = self.raw_buffer.pop(0)
            
            extracted = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', old_obs)
            self.entity_library.update(extracted)

    def _verify_provenance(self, code: str) -> Tuple[bool, str]:
        
        
        potential_entities = re.findall(r"['\"](.*?)['\"]|\b(\d{4,})\b", code)
        flat_entities = [e for sub in potential_entities for e in sub if e]

        
        full_context = " ".join(self.raw_buffer).lower() + " ".join(self.entity_library).lower()

        for entity in flat_entities:
            
            if entity.lower() not in full_context:
                return False, entity
        return True, ""

    def _refine_search_results(self, query: str, raw_text: str, top_k: int = 3) -> str:
        
        if "No results found" in raw_text:
            return raw_text
        clean_text = re.sub(r'\[\d+\]', '', raw_text)
        sentences = re.split(r'(?<=[.!?]) +', clean_text)
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_sentences = []
        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            score = len(query_words.intersection(sent_words))
            scored_sentences.append((score, sent))
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentences = [s[1] for s in scored_sentences[:top_k] if s[0] > 0]
        return " ".join(best_sentences) if best_sentences else " ".join(sentences[:2])

    def execute_python(self, code: str) -> str:
        
        
        is_verified, failed_entity = self._verify_provenance(code)
        if not is_verified:
            return (f"Data Integrity Error: Entity '{failed_entity}' was not found in any prior observations. "
                    f"Action Required: Please verify facts using Wikipedia before assigning variables.")

        
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip() or "Execution successful (no output)."
            else:
                return f"Execution Error:\n{result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (5s limit)."
        except Exception as e:
            return f"Unexpected System Error: {str(e)}"

    def wikipedia_search(self, query):
        try:
            self.verified_entities.add(query.lower())
            
            # 1. 尝试精准获取页面
            try:
                page = wikipedia.page(query, auto_suggest=False)
                return {"content": page.summary[:1200], "status": "precise"}
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                # 2. 【核心改进】暴力回退：获取搜索结果前 3 名的 Snippets
                search_results = wikipedia.search(query)
                if not search_results:
                    return f"Error: 搜索 '{query}' 无任何结果。"
                
                fallback_content = []
                for title in search_results[:3]:
                    try:
                        sum_text = wikipedia.summary(title, sentences=2, auto_suggest=False)
                        fallback_content.append(f"[{title}]: {sum_text}")
                    except:
                        continue
                
                combined_info = "\n".join(fallback_content)
                return {
                    "content": f"精准页面未找到。从相关搜索中提取到碎片信息：\n{combined_info}",
                    "status": "fuzzy"
                }
        except Exception as e:
            return f"Error: 搜索异常 - {str(e)}"

    def dispatch(self, tool_name: str, args: Any) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                if tool_name == "python": args = {"code": args}
                elif tool_name == "wikipedia": args = {"query": args}
                else: return f"Error: Invalid format for tool '{tool_name}'."

        if tool_name == "python":
            code = args.get("code", "") if isinstance(args, dict) else args
            return self.execute_python(str(code))
        elif tool_name == "wikipedia":
            query = args.get("query", "") if isinstance(args, dict) else args
            return self.wikipedia_search(str(query))
        return f"Error: Tool '{tool_name}' not found."