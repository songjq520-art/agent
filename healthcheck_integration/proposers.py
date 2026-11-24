# healthcheck_integration/proposers.py（具名提案函数落点）
#
# 路线①（子进程隔离执行器）要求提案函数是「模块路径:函数名」可
# 点名的真函数：子进程 import 本模块执行，父进程永不 import 这里。
# 本文件只放可信集成代码写的提案函数；模型产出的候选也要先落成
# 这里登记过的函数才许跑，不许临时拼字符串代码执行。
#
# 当前只登记一个安全默认：no_candidate——永远不提代码修改。这与
# 工厂现状一致（repair_paths 默认空、代码自修未启用）；将来真要让
# 提案方产出候选，在这里新增函数并在 factory 的注册表里登记。

def no_candidate(evidence):
    """安全默认提案方：不提任何代码修改（返回 None → no_candidate）。"""
    return None
