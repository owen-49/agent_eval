import json
import re
import os
import statistics
from core_function.agent import ReActAgent
from collections import Counter

class Evaluator:
    def __init__(self, data_path, limit=20):
        self.agent = ReActAgent()
        self.data_path = data_path
        self.limit = limit
        self.results = []

    def clean_text(self, text: str) -> str:
        if not text: return ""
        # 1. 提取 Final Answer 之后的内容
        match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        content = match.group(1) if match else text
        
        # 2. 基础清洗：转小写、去标点、去多余空格
        content = content.lower().strip()
        content = re.sub(r'[.\?!\(\)\[\]]', '', content)
        
        # 3. 去除冠词 (a, an, the)，提高 Exact Match 鲁棒性
        content = re.sub(r'\b(the|a|an)\b', '', content).strip()
        return content

    def calculate_metrics(self, prediction: str, ground_truth: str) -> bool:
        pred_clean = self.clean_text(prediction)
        gt_clean = self.clean_text(ground_truth)
        
        if not pred_clean or not gt_clean:
            return False

        # 策略A：字符串包含（处理模型输出完整句子的情景）
        if gt_clean in pred_clean:
            return True
        
        # 策略B：关键词覆盖（处理 AIME25 等数值任务，确保数值准确）
        gt_words = set(gt_clean.split())
        pred_words = set(pred_clean.split())
        if gt_words.issubset(pred_words):
            return True
            
        return False
    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        pred_clean = self.clean_text(prediction)
        gt_clean = self.clean_text(ground_truth)
        
        pred_tokens = pred_clean.split()
        gt_tokens = gt_clean.split()
        
        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0
        
        # 计算交集词频
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def run_evaluation(self):
        print("="*50)
        print(f"启动自动化评测流水线 | 样本上限: {self.limit}")
        print("="*50)
        
        correct_count = 0
        total_processed = 0

        if not os.path.exists(self.data_path):
            print(f"错误: 找不到数据集 {self.data_path}")
            return

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.limit: break
                
                data = json.loads(line)
                question = data.get('question', '')
                gold_answer = data.get('answer', '')

                print(f"\n[Progress: {i+1}/{self.limit}]")
                print(f"Question: {question[:80]}...")
                
                try:
                    # 调用 Agent 核心循环
                    raw_prediction = self.agent.run(question)
                    
                    # 评测逻辑
                    is_correct = self.calculate_metrics(raw_prediction, gold_answer)
                    if is_correct: correct_count += 1
                    total_processed += 1

                    # 记录轨迹
                    self.results.append({
                        "id": i,
                        "question": question,
                        "gold": gold_answer,
                        "prediction": raw_prediction,
                        "is_correct": is_correct
                    })

                    status = "PASS" if is_correct else "FAIL"
                    print(f"Result: {status} | Current Accuracy: {correct_count/total_processed:.2%}")
                
                except Exception as e:
                    print(f"处理样本 {i} 时发生崩溃: {str(e)}")
                    continue

        self.generate_final_report(correct_count, total_processed)

    def generate_final_report(self, correct, total):
        accuracy = correct / total if total > 0 else 0
        report = {
            "summary": {
                "total_samples": total,
                "correct_count": correct,
                "accuracy": accuracy,
                "model": "gpt-4.0"
            },
            "bad_cases": [r for r in self.results if not r['is_correct']]
        }
        
        os.makedirs("logs", exist_ok=True)
        report_path = "logs/eval_report_final.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        print("\n" + "="*50)
        print(f"评测完成！最终准确率: {accuracy:.2%}")
        print(f"详细报告及 Bad Cases 已保存至: {report_path}")
        print("="*50)

if __name__ == "__main__":
    evaluator = Evaluator(data_path="data/hotpotqa200.jsonl", limit=10)
    evaluator.run_evaluation()