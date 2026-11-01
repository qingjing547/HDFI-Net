"""HDFI-Net验证入口；依据原ultralytics/yolov11/val.py整理。.

原文件实际设置split="test"；本文件使用split="val"，独立测试请运行test.py。
保留原文件的batch=16、NMS IoU=0.60、conf=0.001和workers=4。
"""

from pathlib import Path

from ultralytics import YOLO

# ===== 使用者自行填写：不提供固定的输入、输出路径 =====
WEIGHTS_PATH = ""  # 训练得到的可信best.pt路径
DATA_YAML_PATH = ""  # 数据集YAML路径，需包含val划分
OUTPUT_PROJECT = ""  # 验证结果保存目录
OUTPUT_NAME = ""  # 本次验证名称
DEVICE = ""  # ""自动选择；也可填写"0"或"cpu"

VAL_ARGS = {
    "split": "val",
    "imgsz": 640,
    "batch": 16,
    "iou": 0.70,
    "conf": 0.001,
    "workers": 4,
    "plots": True,
    "save_json": False,
    "exist_ok": False,
}
# 原save_hybrid=False已被当前Ultralytics移除，不再传入这个参数。


def main():
    settings = {
        "WEIGHTS_PATH": WEIGHTS_PATH,
        "DATA_YAML_PATH": DATA_YAML_PATH,
        "OUTPUT_PROJECT": OUTPUT_PROJECT,
        "OUTPUT_NAME": OUTPUT_NAME,
    }
    missing = [key for key, value in settings.items() if not str(value).strip()]
    if missing:
        raise ValueError("请先填写val.py中的：" + "、".join(missing))
    weights = Path(WEIGHTS_PATH).expanduser()
    data = Path(DATA_YAML_PATH).expanduser()
    for path in (weights, data):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
    if weights.suffix.lower() != ".pt":
        raise ValueError("WEIGHTS_PATH应填写训练好的.pt权重，而不是未训练的模型YAML。")
    model = YOLO(str(weights.resolve()), task="detect")
    results = model.val(
        data=str(data.resolve()),
        project=str(Path(OUTPUT_PROJECT).expanduser().resolve()),
        name=OUTPUT_NAME,
        device=DEVICE,
        **VAL_ARGS,
    )
    print(f"验证输出目录：{results.save_dir}")
    return results


if __name__ == "__main__":
    main()
