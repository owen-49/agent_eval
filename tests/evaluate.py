import json
import re
import os
from collections import Counter
from core_function.agent import ReActAgent
from core_function.client import AgentClient

class Evaluator:
    def __init__(self, data_path, limit=10, model_client=None):
        # 核心修复：将 model_client 传递给 ReActAgent
        self.agent = ReActAgent(model_client=model_client)
        self.data_path = data_path
        self.limit = limit
        self.results = []

    def clean_text(self, text: str) -> str:
        if not text: return ""
    
        # 1. 提取标签后的内容，利用正则拦截第一个标点符号或换行符，防止拉取后续解释
        # 这能直接修复 Progress 6 中输出一长串比赛背景的问题
        match = re.search(r"Final Answer:\s*([^。\n\.?!]*)", text, re.IGNORECASE)
        content = match.group(1) if match else text
        
        # 2. 启发式前缀过滤：移除常见的废话开头
        noise_prefixes = ["the answer is", "i conclude that", "it is", "the name is"]
        content = content.lower().strip()
        for prefix in noise_prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
                
        # 3. 标准化：去除冠词与非必要标点
        content = re.sub(r'[^\w\s]', '', content)
        stop_words = {'the', 'a', 'an'}
        tokens = [w for w in content.split() if w not in stop_words]
        
        return " ".join(tokens)

    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.clean_text(prediction).split()
        gt_tokens = self.clean_text(ground_truth).split()
        
        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0
        
        # 计算交集词频[cite: 1]
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_em(self, prediction: str, ground_truth: str) -> bool:
        pred_clean = self.clean_text(prediction)
        gt_clean = self.clean_text(ground_truth)
        
        # 软匹配逻辑：如果标准答案包含在预测中，或预测包含在标准答案中
        if not gt_clean or not pred_clean: return False
        return (gt_clean in pred_clean) or (pred_clean in gt_clean)

    def run_evaluation(self):
        print("="*60)
        print(f"🚀 启动多维自动化评测 | 样本上限: {self.limit}")
        print("="*60)
        
        total_f1 = 0.0
        total_em = 0
        total_samples = 0

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.limit: break
                
                data = json.loads(line)
                question = data.get('question', '')
                gold_answer = data.get('answer', '')

                print(f"\n[Progress: {i+1}/{self.limit}] Question: {question[:60]}...")
                
                try:
                    # 运行 Agent
                    raw_prediction = self.agent.run(question)
                    
                    # 同时计算 EM 和 F1 指标[cite: 1]
                    is_em = self.calculate_em(raw_prediction, gold_answer)
                    f1_val = self.calculate_f1(raw_prediction, gold_answer)
                    
                    total_em += 1 if is_em else 0
                    total_f1 += f1_val
                    total_samples += 1

                    self.results.append({
                        "id": i,
                        "question": question,
                        "gold": gold_answer,
                        "prediction": raw_prediction,
                        "em": is_em,
                        "f1": f1_val
                    })

                    print(f"EM: {'✅' if is_em else '❌'} | F1: {f1_val:.2f} | Running Avg F1: {total_f1/total_samples:.2%}")
                
                except Exception as e:
                    print(f"⚠️ Sample {i} failed: {str(e)}")
                    continue

        self.save_report(total_em, total_f1, total_samples)

    def save_report(self, em_sum, f1_sum, total):
        avg_em = em_sum / total if total > 0 else 0
        avg_f1 = f1_sum / total if total > 0 else 0
        
        report = {
            "summary": {
                "total_samples": total,
                "average_em": avg_em,
                "average_f1": avg_f1,
                "metrics_gap": avg_f1 - avg_em
            },
            "details": self.results
        }
        
        os.makedirs("logs", exist_ok=True)
        with open("logs/eval_report_final.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        print("\n" + "="*60)
        print(f"📊 评测完成！")
        print(f"  > Exact Match (EM): {avg_em:.2%}")
        print(f"  > F1-Score: {avg_f1:.2%}")
        print(f"  > 指标间隙 (F1-EM Gap): {avg_f1 - avg_em:.2%}")
        print("="*60)

if __name__ == "__main__":
    # 1. 首先实例化你的模型客户端
    my_client = AgentClient() 

    # 2. 传入 client，解决 TypeError
    evaluator = Evaluator(
        data_path="data/hotpotqa200.jsonl", 
        limit=20, 
        model_client=my_client
    )
    evaluator.run_evaluation()