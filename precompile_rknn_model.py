import sys
import os
import argparse
from rknn.api import RKNN


def load_model(model_path, npu_id):
    rknn = RKNN()
    devs = rknn.list_devices()
    device_id_dict = {}
    for index, dev_id in enumerate(devs[-1]):
        if dev_id[:2] != 'TS':
            device_id_dict[0] = dev_id
        if dev_id[:2] == 'TS':
            device_id_dict[1] = dev_id
    print('-->loading model : ' + model_path)
    rknn.load_rknn(model_path)
    print('--> Init runtime environment on: ' + device_id_dict[npu_id])
    ret = rknn.init_runtime(device_id=device_id_dict[npu_id], rknn2precompile=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--rknn", type=str, default='', help='weights path')  # from yolov5/models/
    parser.add_argument('-o', "--output", type=str, default='', help='保存name')

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
