import sys
import re

def parse_log(filepath):
    wa_dict = dict()  # {stage: set(idx)}
    ac_dict = dict()
    pattern = re.compile(r"^(\d+)/\d+\s*->\s*(wa|ac)")
    stage = 0
    last_idx = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                idx_str, status = m.group(1), m.group(2)
                idx = int(idx_str)
                # 检测阶段切换
                if last_idx is not None and idx < last_idx:
                    stage += 1
                last_idx = idx
                if status == "wa":
                    wa_dict.setdefault(stage, set()).add(idx)
                elif status == "ac":
                    ac_dict.setdefault(stage, set()).add(idx)
    return wa_dict, ac_dict

def main():
    if len(sys.argv) != 3:
        print("用法: python compare.py log1.log log2.log")
        return

    log1, log2 = sys.argv[1], sys.argv[2]
    wa1_dict, _ = parse_log(log1)
    _, ac2_dict = parse_log(log2)

    total_wa = 0
    total_fixed = 0
    all_fixed = []

    stages = sorted(set(wa1_dict.keys()) | set(ac2_dict.keys()))
    for stage in stages:
        wa1 = wa1_dict.get(stage, set())
        ac2 = ac2_dict.get(stage, set())
        fixed = wa1 & ac2
        total_wa += len(wa1)
        total_fixed += len(fixed)
        all_fixed.extend((stage, idx) for idx in fixed)
        print(f"阶段 {stage}: fix rate: {len(fixed) / len(wa1):.2%} ({len(fixed)} / {len(wa1)})")
        if fixed:
            print("  编号如下：")
            print("  " + ", ".join(str(idx) for idx in sorted(fixed)))
    # 总恢复率
    print(f"\n总恢复率: {total_fixed / total_wa:.2%} ({total_fixed} / {total_wa})")
    if all_fixed:
        print("全部修正编号（格式：阶段号-编号）：")
        print(", ".join(f"{stage}-{idx}" for stage, idx in sorted(all_fixed)))

if __name__ == "__main__":
    main()