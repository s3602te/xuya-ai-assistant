# -*- coding: utf-8 -*-
# tools/calculator.py

import ast
import operator
import re

def calculate_math(expression: str) -> str:
    """
    這是一個安全的數學計算機。
    傳入數學算式 (如: "1250000 / (80000 * 1.15)")，回傳精準的計算結果。
    """
    try:
        # 1. 替換常見的中文與特殊數學符號，確保算式符合 Python 語法
        expression = expression.replace("x", "*").replace("X", "*").replace("÷", "/")
        
        # 【SA 修復】：處理大模型最愛用的次方符號 ^ ，將它轉換為 Python 的 **
        # 注意：避免把正規的二進位運算也換掉，這裡做簡單的全域替換，因為我們是算數學，不是算二進位
        expression = expression.replace("^", "**")
        
        # 清除算式中的千分位逗號 (例如 45,000,000 變成 45000000)
        expression = expression.replace(",", "")
        
        # 2. 安全的節點評估器 (避免執行惡意程式碼)
        def _eval(node):
            if isinstance(node, ast.Num): 
                return node.n
            elif isinstance(node, ast.BinOp):
                op_map = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,   # 支援 ** 次方運算
                    ast.Mod: operator.mod,
                    ast.BitXor: operator.xor # 萬一還是收到 ^ 就當作次方處理(保險起見不開，因為我們已經替換了)
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
    # 測試剛剛失敗的次方與千分位
    test_expr = "500,000 * (1 + 0.07)^15"
    print(f"輸入算式: {test_expr}")
    print("========== 計算結果 ==========")
    print(calculate_math(test_expr))
    print("==============================")