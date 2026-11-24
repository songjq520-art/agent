"""独立检查入口；不动态加载不可信适配器，不提供隐式apply开关。"""
import argparse
import json
from .core import Config, redact
from .checks import Healthcheck


def main():
    parser = argparse.ArgumentParser(description="十二全身体检；独立运行使用未接通适配器")
    parser.add_argument("--config", required=True, help="唯一JSON配置路径")
    parser.add_argument("--weekly", action="store_true", help="每周逾期补跑；需嵌入十二提供租约")
    args = parser.parse_args()
    try:
        check = Healthcheck(Config.load(args.config))
        if args.weekly:
            result = check.weekly()
            print(json.dumps(redact(result), ensure_ascii=False))
            return 0 if result.get("ran") else 2
        path, report = check.report()
        print(json.dumps(redact({"report": str(path), "findings": report["findings"]}), ensure_ascii=False))
        return 1 if any(f["status"] == "异常" for f in report["findings"]) else 0
    except Exception as exc:
        print(json.dumps(redact({"exit_status": "configuration_error", "error": str(exc)}), ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
