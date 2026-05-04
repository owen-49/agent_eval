import json
import re
from collections import Counter
from core_function.agent import ReActAgent

class Evaluator:
    def __init__(self, data_path, limit=10, model_client=None):
        # 传入 model_client 以支持解耦的 LLM 调用
        self.agent = ReActAgent(model_client=model_client)
        self.data_path = data_path
        self.limit = limit
        self.results = []

    def clean_text(self, text: str) -> str:
        """
        学术级文本清洗：去标签、小写化、去冠词。
        用于降低 Exact Match 的评测噪声。
        """
        if not text: return ""
        # 仅提取 Final Answer 后的核心内容
        match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        content = match.group(1) if match else text
        content = content.lower().strip()
        # 去除标点与冠词
        content = re.sub(r'[^\w\s]', '', content)
        content = re.sub(r'\b(the|a|an)\b', '', content).strip()
        return content

    def calculate_metrics(self, prediction: str, ground_truth: str):
        """
        计算多维指标：包含 EM 和 Token-level F1[cite: 2]。
        """
        pred_clean = self.clean_text(prediction)
        gt_clean = self.clean_text(ground_truth)
        
        # 1. Exact Match (EM)
        is_em = gt_clean in pred_clean if gt_clean else False
        
        # 2. F1-Score (基于词级的召回率与精确率)[cite: 2]
        pred_tokens = pred_clean.split()
        gt_tokens = gt_clean.split()
        
        if not pred_tokens or not gt_tokens:
            f1 = 1.0 if pred_tokens == gt_tokens else 0.0
        else:
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                precision = 1.0 * num_same / len(pred_tokens)
                recall = 1.0 * num_same / len(gt_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
        
        return is_em, f1

    def run_evaluation(self):
        print("="*50)
        print(f"🚀 启动硬约束评测流水线 | 样本上限: {self.limit}")
        print("="*50)
        
        total_em = 0
        total_f1 = 0.0
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.limit: break
                data = json.loads(line)
                question = data.get('question')
                gold = data.get('answer')
                
                print(f"\n[Progress: {i+1}/{self.limit}]")
                try:
                    # 运行具备硬约束校验的 Agent
                    prediction = self.agent.run(question)
                    is_em, f1 = self.calculate_metrics(prediction, gold)
                    
                    total_em += 1 if is_em else 0
                    total_f1 += f1
                    
                    print(f"EM: {'✅' if is_em else '❌'} | F1: {f1:.2f}")
                    
                    self.results.append({
                        "id": i,
                        "em": is_em,
                        "f1": f1,
                        "pred": prediction,
                        "gold": gold
                    })
                except Exception as e:
                    print(f"崩溃处理: {str(e)}") # 捕捉处理异常，提升系统弹性
                    continue

        self.print_summary(total_em, total_f1, i + 1)

    def print_summary(self, em, f1, total):
        avg_em = em / total
        avg_f1 = f1 / total
        print("\n" + "="*50)
        print(f"📊 最终统计 (n={total})")
        print(f"Average EM: {avg_em:.2%}")
        print(f"Average F1: {avg_f1:.2%}")
        print(f"Metrics Gap (F1-EM): {avg_f1 - avg_em:.2%}") # 指标间隙分析[cite: 2]
        print("="*50)