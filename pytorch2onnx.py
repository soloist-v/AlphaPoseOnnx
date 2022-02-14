import argparse
import copy
import json
import os
from typing import Tuple, Dict, Any
import torch.nn as nn
import torch
import torch.fx as fx


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert (isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.
        if node.op != 'call_module':  # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


def save_torch_script(weight, cfg, torch_script_name, c, h, w):
    from alphapose.models.fastpose import FastPose
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    cfg = update_config(cfg)
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()
    # 输入数据形状
    dummy_input = torch.randn(1, c, h, w, device='cpu')
    # ----------export onnx--------------
    # torch.onnx.export(model, dummy_input, save_path, opset_version=12)
    # print("onnx save success.")
    # ---------torch script-------------
    print(torch_script_name)
    model = fuse(model)
    model.eval()
    ts = torch.jit.trace(model, dummy_input, strict=False)
    # d = {"shape": dummy_input.shape}
    # extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    ts.save(torch_script_name)
    print("torch script saved.")


def torch_script_to_onnx(weight, save_path, c, h, w, device="cpu"):
    # 输入数据形状
    dummy_input = torch.randn(1, c, h, w, device=device)
    model = torch.jit.load(weight, device)
    # ----------export onnx--------------
    torch.onnx.export(model, dummy_input, save_path, opset_version=10)
    print("onnx save success.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-i", type=str, help="AlphaPose weights filepath")
    parser.add_argument("--onnx", "-o", type=str, default="", help="onnx save name")
    parser.add_argument("--cfg", type=str, default="configs/256x192_res50_lr1e-3_1x.yaml", help="AlphaPose config")
    opt = parser.parse_args()
    if opt.onnx == "":
        base_name = os.path.splitext(opt.weights)[0]
        opt.onnx = f"{base_name}.onnx"
    return opt


def main():
    opt = parse_args()
    ts = "fastpose_res50.torchscript.pt"
    save_torch_script(opt.weights, opt.cfg, ts, 3, 256, 192)
    torch_script_to_onnx(ts, opt.onnx, 3, 256, 192)


if __name__ == '__main__':
    main()
