import argparse
import os
from rknn.api import RKNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='weights path')
    parser.add_argument('--rknn', type=str, default='', help='保存路径')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dataset', type=str, default="./dataset.txt", help='dataset txt')
    parser.add_argument('--platform', type=str, default="rk3399pro", help='target platform')
    opt = parser.parse_args()
    print("options:\n\t", opt)
    ONNX_MODEL = opt.onnx
    if opt.rknn:
        RKNN_MODEL = opt.rknn
    else:
        RKNN_MODEL = "%s.rknn" % os.path.splitext(ONNX_MODEL)[0]
    rknn = RKNN()  # verbose=True
    print('--> config model')
    # rknn.config(batch_size=opt.batch_size,
    #             target_platform=opt.platform,
    #             mean_values=[[0.406 * 255, 0.457*255, 0.480*255]],
    #             std_values=[[255.0, 255.0, 255.0]])
    rknn.config(batch_size=opt.batch_size,
                target_platform=opt.platform,
                mean_values=[[0, 0, 0]],
                std_values=[[1, 1, 1]])
    # Load model
    print('--> Loading model')
    print('onnx model path:', ONNX_MODEL)
    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, "Load onnx failed!"
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=opt.dataset)  # pre_compile=True
    assert ret == 0, "Build onnx failed!"
    # Export model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    assert ret == 0, "Export %s.rknn failed!" % opt.rknn
    print("rknn export success, saved as %s" % RKNN_MODEL)
    print('done')
