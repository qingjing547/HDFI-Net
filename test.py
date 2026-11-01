"""HDFI-Net测试入口；依据原ultralytics/yolov11/test.py整理。.

使用独立测试集，输出权重文件大小、Precision、Recall、F1、mAP50、
mAP50-95以及基于预处理、推理、后处理计时的Pipeline FPS。
"""

from pathlib import Path

from ultralytics import YOLO

# ===== 使用者自行填写：不提供固定的输入、输出路径 =====
WEIGHTS_PATH = ""  # 训练得到的可信best.pt路径
DATA_YAML_PATH = ""  # 数据集YAML路径，需包含test划分
OUTPUT_PROJECT = ""  # 测试结果保存目录
OUTPUT_NAME = ""  # 本次测试名称
DEVICE = ""  # ""自动选择；也可填写"0"或"cpu"

TEST_ARGS = {
    "split": "test",
    "imgsz": 640,
    "batch": 4,
    "conf": 0.001,  # 将原验证器默认的AP评价阈值明确写出
    "iou": 0.70,  # NMS IoU，采用原test.py对应的框架默认值
    "workers": 8,
    "plots": True,
    "save_json": False,
    "exist_ok": False,
}


def print_metrics(results, model_size_mb):
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    speed = results.speed or {}
    time_per_image_ms = sum(float(speed.get(stage, 0.0)) for stage in ("preprocess", "inference", "postprocess"))
    fps = 1000.0 / time_per_image_ms if time_per_image_ms > 0 else 0.0
    print("\n" + "=" * 64)
    print("HDFI-Net测试指标")
    print(f"权重文件大小：{model_size_mb:.2f} MB（字节数除以1024²）")
    print(f"Precision：  {precision:.4f}")
    print(f"Recall：     {recall:.4f}")
    print(f"F1-score：   {f1_score:.4f}")
    print(f"mAP50：      {float(results.box.map50):.4f}")
    print(f"mAP50-95：   {float(results.box.map):.4f}")
    print(f"Pipeline延迟：{time_per_image_ms:.2f} ms/图")
    print(f"Pipeline FPS：{fps:.2f}")
    print("F1为平均Precision与平均Recall的调和平均。")
    print("FPS仅对应当前评价流程计时，不包含文件读取、摄像头、界面或网络开销。")
    print("=" * 64)


def main():
    settings = {
        "WEIGHTS_PATH": WEIGHTS_PATH,
        "DATA_YAML_PATH": DATA_YAML_PATH,
        "OUTPUT_PROJECT": OUTPUT_PROJECT,
        "OUTPUT_NAME": OUTPUT_NAME,
    }
    missing = [key for key, value in settings.items() if not str(value).strip()]
    if missing:
        raise ValueError("请先填写test.py中的：" + "、".join(missing))
    weights = Path(WEIGHTS_PATH).expanduser()
    data = Path(DATA_YAML_PATH).expanduser()
    for path in (weights, data):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
    if weights.suffix.lower() != ".pt":
        raise ValueError("WEIGHTS_PATH应填写训练好的.pt权重，而不是未训练的模型YAML。")
    model_size_mb = weights.stat().st_size / (1024 * 1024)
    model = YOLO(str(weights.resolve()), task="detect")
    results = model.val(
        data=str(data.resolve()),
        project=str(Path(OUTPUT_PROJECT).expanduser().resolve()),
        name=OUTPUT_NAME,
        device=DEVICE,
        **TEST_ARGS,
    )
    print_metrics(results, model_size_mb)
    print(f"测试输出目录：{results.save_dir}")
    return results


if __name__ == "__main__":
    main()
