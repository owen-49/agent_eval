import unittest
from core_function.tools import ToolManager

class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.tm = ToolManager()

    def test_python_success(self):
        """测试代码执行的正确性"""
        res = self.tm.dispatch("python", {"code": "print(10 + 20)"})
        self.assertEqual(res.strip(), "30")

    def test_python_security_timeout(self):
        """测试安全沙箱：死循环拦截"""
        res = self.tm.dispatch("python", {"code": "while True: pass"})
        self.assertIn("timed out", res)

    def test_python_error_feedback(self):
        """测试错误反馈机制：是否返回 Traceback 以供 Agent 纠错"""
        res = self.tm.dispatch("python", {"code": "print(undefined_variable)"})
        self.assertIn("NameError", res)

    def test_token_refresh_logic(self):
        """测试运行时自愈：Token 刷新机制"""
        # 强制设置一个会导致失败的状态
        self.tm._session_token = "fail_token"
        res = self.tm.dispatch("wikipedia", {"query": "Quantum Physics"})
        self.assertIn("System Alert", res)
        self.assertIn("refreshed", res)

if __name__ == "__main__":
    unittest.main()