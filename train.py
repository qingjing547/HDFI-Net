"""HDFI-Net训练入口；参数依据原ultralytics/yolov11/train.py整理。

运行前填写下面的输入、输出路径。MODEL_YAML_PATH应指向本项目的
ultralytics/cfg/models/HDFI-Net/HDFI-Net.yaml，不加载预训练权重。
Windows路径建议使用原始字符串，例如r"D:\\data\\cotton.yaml"。
"""

from pathlib import Path

from ultralytics import YOLO


# ===== 使用者自行填写：不提供固定的输入、输出路径 =====
MODEL_YAML_PATH = "HDFI-Net.yaml"  # HDFI-Net.yaml的路径
DATA_YAML_PATH = ""  # 数据集YAML路径
OUTPUT_PROJECT = ""  # 结果保存目录
OUTPUT_NAME = ""  # 本次实验名称
DEVICE = ""  # ""自动选择；也可填写"0"或"cpu"

# ===== 保留原训练脚本的参数；按自己的硬件修改batch、workers等 =====
TRAIN_ARGS = {
    "epochs": 500,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "amp": False,
    "cache": False,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 10,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.0,
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "dropout": 0.0,
    "patience": 80,
    "save": True,
    "save_period": 20,
    "exist_ok": False,
    "val": True,
    "plots": True,
    "verbose": True,
    "deterministic": True,
    "seed": 0,
    "fraction": 1.0,
    "single_cls": False,
    "rect": False,
    "resume": False,
    "multi_scale": False,
    "profile": False,
    "split": "val",
    "save_json": False,
    "iou": 0.7,
    "max_det": 300,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "overlap_mask": True,
    "mask_ratio": 4,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "close_mosaic": 10,
    "pose": 12.0,
    "kobj": 1.0,
    "pretrained": False,
}


def main():
    settings = {
        "MODEL_YAML_PATH": MODEL_YAML_PATH,
        "DATA_YAML_PATH": DATA_YAML_PATH,
        "OUTPUT_PROJECT": OUTPUT_PROJECT,
        "OUTPUT_NAME": OUTPUT_NAME,
    }
    missing = [key for key, value in settings.items() if not str(value).strip()]
    if missing:
        raise ValueError("请先填写train.py中的：" + "、".join(missing))
    model_path = Path(MODEL_YAML_PATH).expanduser()
    data_path = Path(DATA_YAML_PATH).expanduser()
    for path in (model_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
    if model_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("从头训练时MODEL_YAML_PATH应填写模型YAML，而不是.pt权重。")
    model = YOLO(str(model_path.resolve()), task="detect")
    results = model.train(
        data=str(data_path.resolve()),
        project=str(Path(OUTPUT_PROJECT).expanduser().resolve()),
        name=OUTPUT_NAME,
        device=DEVICE,
        **TRAIN_ARGS,
    )
    print(f"训练输出目录：{model.trainer.save_dir}")
    print(f"最佳验证检查点：{model.trainer.best}")
    return results


if __name__ == "__main__":
    main()
