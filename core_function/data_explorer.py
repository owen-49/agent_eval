import json
import statistics
from collections import Counter

def explore_hotpot_data(file_path):
    questions = []
    answers = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data.get('question', ''))
                answers.append(data.get('answer', ''))
    except FileNotFoundError:
        print(f"Error: 找不到文件 {file_path}，请检查路径。")
        return

    # 1. 基础统计
    total_count = len(questions)
    q_lengths = [len(q) for q in questions]
    
    print("="*30)
    print(f"项目B：HotpotQA 数据集初步分析报告")
    print("="*30)
    print(f"总样本数: {total_count}")
    print(f"问题平均长度 (字符): {statistics.mean(q_lengths):.2f}")
    print(f"问题长度中位数: {statistics.median(q_lengths)}")
    
    
    reasoning_keywords = ['both', 'and', 'which', 'who', 'first', 'second', 'between']
    keyword_counts = Counter()
    for q in questions:
        for word in reasoning_keywords:
            if word in q.lower():
                keyword_counts[word] += 1

    print("\n[推理特征统计]")
    for word, count in keyword_counts.items():
        print(f" - 关键词 '{word}' 出现频率: {count/total_count:.2%}")

    
    print("\n[样本深度观察 - 模拟 Agent 逻辑]")
    for i in range(min(2, total_count)):
        print(f"\nExample #{i+1}:")
        print(f"Q: {questions[i]}")
        print(f"A: {answers[i]}")
        print("-" * 10)

if __name__ == "__main__":
    DATA_PATH = "data/hotpotqa200.jsonl"
    explore_hotpot_data(DATA_PATH)