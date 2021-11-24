import torch
import onnx
from alphapose.models.fastpose import FastPose
from alphapose.models import builder
from alphapose.utils.config import update_config


def save_onnx(model, filepath, c, h, w):
    # 输入数据形状
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx.export(model, dummy_input, filepath, opset_version=11)


cfg = update_config("./256x192_res50_lr1e-3_1x.yaml")
print(cfg)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load("./fast_res50_256x192.pth"))
pose_model.eval()
print(pose_model)
pose_model = pose_model.cuda()
save_onnx(pose_model, filepath="./fastpose_ret50.onnx", c=3, h=256, w=192)
