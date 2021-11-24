import argparse
import os.path
import torch
import onnx
from alphapose.models.fastpose import FastPose
from alphapose.models import builder
from alphapose.utils.config import update_config


def export_onnx(model, filepath, c, h, w):
    # 输入数据形状
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx.export(model, dummy_input, filepath, opset_version=11)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-i", type=str, help="AlphaPose weights filepath")
    parser.add_argument("--onnx", "-o", type=str, default="", help="onnx save name")
    parser.add_argument("--cfg", type=str, default="./256x192_res50_lr1e-3_1x.yaml", help="AlphaPose config")
    opt = parser.parse_args()
    if opt.onnx == "":
        base_name = os.path.splitext(opt.weights)[0]
        opt.onnx = f"{base_name}.onnx"
    return opt


def main():
    opt = parse_args()
    cfg = update_config(opt.cfg)
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(opt.weights))
    pose_model.eval()
    pose_model = pose_model.cuda()
    export_onnx(pose_model, opt.onnx, c=3, h=256, w=192)


if __name__ == '__main__':
    main()
