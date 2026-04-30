from core_function.agent import ReActAgent

def test_multihop_reasoning():
    """
    测试集成：验证 Agent 是否能完成多步搜索与信息聚合
    """
    agent = ReActAgent()
    
    # 这是一个典型的 HotpotQA 式多跳问题
    # 逻辑：需要先搜 Truman Sports Complex -> 找到两个球场 -> 确定另一个是 Kauffman Stadium
    question = "In the 1973 NFL season, the Pro Bowl took place at what football stadium that is part of the Truman Sports Complex, along with what other stadium?"
    
    print("="*50)
    print(f"🧪 正在测试多跳推理能力...")
    result = agent.run(question)
    
    print("\n" + "="*50)
    print(f"✅ 任务结束！最终输出：\n{result}")

if __name__ == "__main__":
    test_multihop_reasoning()