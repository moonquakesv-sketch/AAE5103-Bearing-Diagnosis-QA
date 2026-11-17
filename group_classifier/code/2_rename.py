# -*- coding: utf-8 -*-
"""
批量给 .npy 文件加 {故障类型}_HP{负载}_ 前缀。
目录结构：
  CWRU_HP0/
    B007_0/seg_0000.npy       ->  B007_0/B007_HP0_seg_0000.npy
    OR007@6_0/seg_0001.npy    ->  OR007@6_0/OR007@6_HP0_seg_0001.npy
    ...
"""
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "Data" / "HP_0_processed"

DRY_RUN   = False    # True=只打印不实际改名；确认无误后改 False 再跑一次
COPY_MODE = False   # True=保留原文件、复制一份新名；False=原地重命名


def parse_folder(name: str):
    """B007_0 -> ('B007', 0)；OR007@6_0 -> ('OR007@6', 0)；无后缀则返回 (name, None)"""
    m = re.match(r"^(.+?)_(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return name, None


def get_hp(base_path: Path, folder_name: str):
    """优先父目录 CWRU_HPx，其次文件夹末尾 _x"""
    m = re.match(r"^CWRU_HP(\d+)$", base_path.name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    _, tail = parse_folder(folder_name)
    return tail


def main():
    total = renamed = skipped = conflict = 0

    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("seg_*.npy"))
        if not files:
            continue

        fault, _ = parse_folder(folder.name)
        hp = get_hp(BASE, folder.name)
        if hp is None:
            print(f"[跳过] {folder.name}: 无法解析 HP")
            continue

        prefix = f"{fault}_HP{hp}_"
        print(f"\n[{folder.name}]  前缀: {prefix}   ({len(files)} 个文件)")

        for f in files:
            total += 1
            if f.name.startswith(prefix):
                skipped += 1
                continue

            new_path = f.with_name(prefix + f.name)
            if new_path.exists():
                print(f"  [冲突] {new_path.name} 已存在，跳过 {f.name}")
                conflict += 1
                continue

            if DRY_RUN:
                print(f"  [dry] {f.name}  ->  {new_path.name}")
            elif COPY_MODE:
                shutil.copy2(f, new_path)
            else:
                f.rename(new_path)
            renamed += 1

    print(f"\n完成：扫描 {total}，改名/复制 {renamed}，跳过 {skipped}，冲突 {conflict}")
    if DRY_RUN:
        print("⚠ DRY_RUN=True，未实际改动。确认无误后改为 False 再执行一次。")


if __name__ == "__main__":
    main()