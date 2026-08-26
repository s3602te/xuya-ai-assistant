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
                    # 【SA v3.3 新增】整數除法。
                    # 分配題（「25顆分給7個人，每人幾顆」）需要的是整除 25 // 7 = 3，
                    # 而不是 25 / 7 = 3.571。少了這個運算子，
                    # 算盤法師就算翻譯正確也會在這裡拋出「不支援的運算」而失敗。
                    ast.FloorDiv: operator.floordiv,
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
# 【SA v4.0 新增】鏈式推導模式 (calculate_script)
# ============================
# 為什麼需要這個？
#
# 舊的 calculate_math 一次只能算一條獨立算式，這對「數據依賴型」的多步推理無能為力。
# 實測翻車案例（文具福袋題）：
#   「有75本筆記本、52支鋼筆、38瓶墨水，平均分裝成8個福袋，各剩多少？
#     再把所有剩下的物品，每3件換1張貼紙，可以換幾張？」
#
# 第二小題需要「先算出三個餘數 3、4、6，加起來得到 13，再除以 3」。
# 但規劃節點在開場時根本不知道餘數是多少，無法預先排出 13 // 3 這條算式；
# 而算盤法師單次呼叫也只能做一次翻譯。
# 結果模型只能用猜的，把「每袋幾本」(9+6+4) 當成「剩下幾本」拿去除以 3，
# 算出 6 張貼紙 —— 正解其實是 4 張。
#
# 解法：讓模型寫一小段【只有數學的腳本】，用變數表達步驟之間的依賴關係：
#     r1 = 75 % 8
#     r2 = 52 % 8
#     r3 = 38 % 8
#     total = r1 + r2 + r3
#     stickers = total // 3
#     leftover = total % 3
#
# 為什麼選這條路，而不是讓 Agent 跑多輪 ReAct 迴圈？
#   因為多輪迴圈 = 每一輪都要模型判斷「做完了沒」，而模型判斷正是本專案所有錯誤的來源。
#   腳本的作法只需要模型寫【一次】，之後的執行完全是確定的、可驗證的、可重現的。
#   變數本身就是依賴關係，不需要模型在多輪之間「記得」中間結果。
#
# 安全性：沿用同一套 AST 白名單，只多開放「變數指派」與「讀取自己定義過的變數」。
# 仍然禁止 import、屬性存取、下標、迴圈、條件式、以及白名單以外的任何函式呼叫。
# ============================
def calculate_script(script: str) -> dict:
    """
    執行一小段只含數學的推導腳本，回傳每一步的結果。

    回傳格式：
      {
        "ok": bool,
        "steps": [(變數名, 算式原文, 結果), ...],   # 依執行順序
        "vars": {變數名: 值},
        "message": str                              # 失敗時的錯誤說明
      }
    """
    try:
        # 1. 沿用單條算式的前處理（中文符號、次方、千分位逗號）
        script = script.replace("÷", "/").replace("^", "**")
        script = re.sub(r'(?<=[\d\)])\s*[xX×]\s*(?=[\d\(])', '*', script)
        script = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', script)

        tree = ast.parse(script, mode="exec")

        env = {}
        steps = []

        def _eval(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise TypeError(f"不支援的常數型別: {type(node.value)}")
                return node.value
            elif isinstance(node, ast.Name):
                # 【SA v4.0】只准讀取這段腳本自己定義過的變數，
                # 讀不存在的名字一律報錯（避免模型憑空引用不存在的中間結果）
                if node.id not in env:
                    raise NameError(f"使用了未定義的變數: {node.id}")
                return env[node.id]
            elif isinstance(node, ast.BinOp):
                op_map = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Pow: operator.pow,
                    ast.Mod: operator.mod,
                }
                return op_map[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                op_map = {ast.UAdd: operator.pos, ast.USub: operator.neg}
                return op_map[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Call):
                func_name = getattr(node.func, "id", None)
                if func_name not in _ALLOWED_FUNCS:
                    raise ValueError(f"不允許呼叫函式: {func_name}")
                if node.keywords:
                    raise ValueError("不允許使用關鍵字參數")
                int_args = []
                for a in [_eval(x) for x in node.args]:
                    if isinstance(a, float) and not a.is_integer():
                        raise ValueError(f"排列組合/階乘的參數必須是整數，收到: {a}")
                    int_args.append(int(a))
                return _ALLOWED_FUNCS[func_name](*int_args)
            else:
                raise TypeError(f"不支援的運算節點: {type(node).__name__}")

        # 2. 逐行執行：只允許「單一變數指派」與「單獨的算式」
        if len(tree.body) > 20:
            raise ValueError("腳本超過 20 行，過於複雜")

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise ValueError("只允許『單一變數 = 算式』的形式")
                name = stmt.targets[0].id
                val = _eval(stmt.value)
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                env[name] = val
                steps.append((name, ast.unparse(stmt.value), val))
            elif isinstance(stmt, ast.Expr):
                val = _eval(stmt.value)
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                steps.append((None, ast.unparse(stmt.value), val))
            else:
                # 明確擋掉 import / for / if / while / def 等一切非算術語句
                raise ValueError(f"不支援的語句類型: {type(stmt).__name__}")

        if not steps:
            raise ValueError("腳本沒有任何可執行的算式")

        return {"ok": True, "steps": steps, "vars": env, "message": ""}

    except Exception as e:
        return {"ok": False, "steps": [], "vars": {}, "message": str(e)}


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