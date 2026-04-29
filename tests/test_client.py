from core_function.client import AgentClient

def test_basic_communication():
    """测试基础的请求与响应"""
    client = AgentClient()
    prompt = "State the goal of the HotpotQA dataset in one sentence."
    
    print("正在发送测试请求...")
    response = client.request_llm([{"role": "user", "content": prompt}])
    
    if response:
        print(f"测试成功！模型回复：\n{response}")
    else:
        print("测试失败：未能获取模型回复。")

if __name__ == "__main__":
    test_basic_communication()