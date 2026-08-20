# -*- coding: utf-8 -*-
# tools/calculator.py

import ast
import operator
import math
import re

# 【SA 新增】：排列組合/階乘函式白名單，只允許呼叫這三個安全函式，
# 避免任意 Python 函式被塞進 eval 造成安全風險
_ALLOWED_FUNCS = {
    "comb": math.comb,       # 組合數 comb(n, k)
    "perm": math.perm,       # 排列數 perm(n, k)
    "factorial": math.factorial,  # 階乘 factorial(n)
}


def calculate_math(expression: str) -> str:
    """
    這是一個安全的數學計算機。
    傳入數學算式 (如: "1250000 / (80000 * 1.15)")，回傳精準的計算結果。
    【SA 新增】：也支援排列組合與階乘，例如 "comb(5, 2)"、"perm(5, 2)"、"factorial(5)"。
    """
    try:
        # 1. 替換常見的中文與特殊數學符號，確保算式符合 Python 語法
        # 【SA v2 修正】：舊版是無條件 expression.replace("x", "*")，
        # 這會把函式名稱或任何含 x 的字元一併炸掉(目前白名單函式雖然沒有 x，
        # 但只要日後加一個 max/exp 之類的函式就會立刻出事)。
        # 改成只在「數字或右括號」與「數字或左括號」之間的 x 才視為乘號。
        expression = re.sub(r'(?<=[\d\)])\s*[xX×]\s*(?=[\d\(])', '*', expression)
        expression = expression.replace("÷", "/")

        # 【SA 修復】：處理大模型最愛用的次方符號 ^ ，將它轉換為 Python 的 **
        expression = expression.replace("^", "**")

        # 清除算式中的千分位逗號 (例如 45,000,000 變成 45000000)
        # 【SA 注意】：comb(5, 2) 這種函式呼叫也用逗號分隔參數，
        # 這裡的清除規則只會拿掉「數字之間」的千分位逗號，不會動到函式參數之間的逗號。
        expression = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', expression)

        # 2. 安全的節點評估器 (避免執行惡意程式碼)
        def _eval(node):
            # 【SA v2 修正】：ast.Num 從 Python 3.8 起被標記為 deprecated，
            # 3.12 開始會噴 DeprecationWarning、未來版本會直接移除。
            # 改用 ast.Constant，並明確擋掉非數字的常數(字串、布林等)。
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise TypeError(f"不支援的常數型別: {type(node.value)}")
                return node.value
            elif isinstance(node, ast.BinOp):
                op_map = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,   # 支援 ** 次方運算
                    ast.Mod: operator.mod,
                    ast.BitXor: operator.xor
                }
                left = _eval(node.left)
                right = _eval(node.right)
                op = op_map[type(node.op)]
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):
                op_map = {
                    ast.UAdd: operator.pos,
                    ast.USub: operator.neg
                }
                operand = _eval(node.operand)
                op = op_map[type(node.op)]
                return op(operand)
            elif isinstance(node, ast.Call):
                # 【SA 新增】：只允許呼叫白名單內的排列組合/階乘函式，其餘一律拒絕
                func_name = getattr(node.func, "id", None)
                if func_name not in _ALLOWED_FUNCS:
                    raise ValueError(f"不允許呼叫函式: {func_name}")
                # 【SA v2 新增】：擋掉關鍵字參數與 *args 展開，避免繞過白名單檢查
                if node.keywords:
                    raise ValueError("不允許使用關鍵字參數")
                args = [_eval(arg) for arg in node.args]
                int_args = []
                for a in args:
                    if isinstance(a, float) and not a.is_integer():
                        raise ValueError(f"排列組合/階乘的參數必須是整數，收到: {a}")
                    int_args.append(int(a))
                return _ALLOWED_FUNCS[func_name](*int_args)
            else:
                raise TypeError(f"不支援的運算節點: {type(node)}")

        # 3. 解析算式並計算
        tree = ast.parse(expression, mode='eval')
        result = _eval(tree.body)

        # 4. 格式化輸出
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"計算成功！算式 '{expression}' 的結果為：{result}"

    except Exception as e:
        return f"計算失敗！請檢查算式格式是否有誤。錯誤訊息：{e}"


# ============================
# 單元測試區塊 (僅直接執行此檔案時觸發)
# ============================
if __name__ == "__main__":
    print("🧮 正在啟動數學計算機測試...\n")
    test_expr = "500,000 * (1 + 0.07)^15"
    print(f"輸入算式: {test_expr}")
    print("========== 計算結果 ==========")
    print(calculate_math(test_expr))
    print("==============================")

    # 【SA 新增】測試排列組合與階乘
    for expr in ["comb(5, 2)", "perm(5, 2)", "factorial(5)",
                 "comb(4,1) * comb(2,1) + comb(4,1) * comb(3,1)",
                 "634 - 508", "12 x 8"]:
        print(f"\n輸入算式: {expr}")
        print(calculate_math(expr))