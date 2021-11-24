import sys
import os
import argparse
import warnings


def load_model(model_path, npu_id):
    from rknn.api import RKNN
    if isinstance(npu_id, str):
        npu_id = int(npu_id)
    rknn = RKNN()
    devs = rknn.list_devices()
    dev_id: str
    src_devices = []
    ts_devices = []
    for index, dev_id in enumerate(devs[-1]):
        if dev_id.startswith("TS"):
            ts_devices.append(dev_id)
        else:
            src_devices.append(dev_id)
    all_devices = src_devices + ts_devices
    if len(all_devices) == 1:
        npu_id = 0
        warnings.warn("当前只有1个npu设备，强制加载到设备0", UserWarning)
    else:
        assert npu_id < len(all_devices), "npu_id 超出设备数量索引值."
    print("--> All devices: ", all_devices)
    print('-->loading model : ' + model_path)
    rknn.load_rknn(model_path)
    print('-->Init runtime environment on: ' + all_devices[npu_id])
    ret = rknn.init_runtime(device_id=all_devices[npu_id] if len(all_devices) > 1 else None, rknn2precompile=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--rknn", type=str, default='', help='weights path')  # from yolov5/models/
    parser.add_argument('-o', "--outputs", type=str, default='', help='保存name')

    opt = parser.parse_args()
    orig_rknn = opt.rknn
    out_rknn = opt.output
    if out_rknn == "":
        out_rknn = "%s_precompile.rknn" % os.path.splitext(orig_rknn)[0]

    # Create RKNN object
    print('--> Loading RKNN model')
    rknn = load_model(orig_rknn, 0)

    ret = rknn.export_rknn_precompile_model(out_rknn)

    rknn.release()
