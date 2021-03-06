import time
import cv2
import numpy as np
import onnxruntime
import torch
from torch import Tensor
from numpy import ndarray
from pPose_nms import pose_nms
from bbox import _box_to_center_scale, _center_scale_to_box
from transforms import get_affine_transform, im_to_torch, heatmap_to_coord_simple
from const import joint_pairs, human_keypoint_labels
from plot import plot_poses, plot_pose


def flip(x: "Tensor"):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1
    return x.flip(dims=(dim,))


def flip_heatmap(heatmap: "ndarray", shift=False):
    """Flip pose heatmap according to joint pairs.
    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
        List of joint pairs.
    shift : bool
        Whether to shift the outputs.
    Returns
    -------
    numpy.ndarray
        Flipped heatmap.
    """
    assert (heatmap.ndim == 3 or heatmap.ndim == 4)
    heatmap = torch.from_numpy(heatmap)
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        print(dim0, dim1)
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out


def get_keypoints(ren_indexes, boxes, hm_data: "Tensor", cropped_boxes, fn=0):
    # 暂时先构造的scores和ids,作为后续pose_nms的参数
    min_box_area = 0
    scores = torch.ones(len(boxes))
    ids = torch.zeros(scores.shape)
    eval_joints = list(range(17))
    norm_type = None
    hm_size = [64, 48]
    if boxes is None or len(boxes) == 0:
        return None
    else:
        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm_data.dim() == 4
        if hm_data.size()[1] == 136:
            eval_joints = [*range(0, 136)]
        elif hm_data.size()[1] == 26:
            eval_joints = [*range(0, 26)]
        pose_coords = []
        pose_scores = []

        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = heatmap_to_coord_simple(hm_data[i][eval_joints], bbox, hm_shape=hm_size,
                                                             norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(boxes, scores, ids, preds_img,
                                                                         preds_scores, min_box_area)
        _result = []
        for k in range(len(scores)):
            _result.append(
                {
                    'keypoints': preds_img[k],
                    'kp_score': preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx': ids[k],
                    'bbox': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]],
                    'index': ren_indexes[k]
                }
            )

        result = {
            'img': str(fn) + '.jpg',
            'result': _result
        }
    return result


def _transform(src_img, bbox, input_size=(256, 192), aspect_ratio=192 / 256):
    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
    scale = scale * 1.0
    inp_h, inp_w = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(src_img, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)
    # img = im_to_torch(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = img.astype(np.float32)
    img /= 255
    img[0] += -0.406
    img[1] += -0.457
    img[2] += -0.480
    return img, bbox


def get_pose_boxes(img, pose, need_keypoints='all'):
    h, w = img.shape[:2]
    labels = []
    boxes = []
    poses = {}
    if pose is not None and len(pose['result']) > 0:
        kp_num = len(pose['result'][0]['keypoints'])
        assert kp_num == 17
        for human in pose['result']:
            part_line = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            ren_src_index = human['index']
            # cur_pose = {}
            # 颈部关键点通过计算得出
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            # 关键点
            for n in range(kp_scores.shape[0]):
                if (need_keypoints != 'all' and human_keypoint_labels[n] not in need_keypoints) \
                        or kp_scores[n] <= 0.4:  # 移除不检测或置信度过低的关键点
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                # cur_pose[human_keypoint_labels[n]] = (cor_x, cor_y)
                boxes.append([max(0, cor_x - 15), max(0, cor_y - 20), min(w, cor_x + 15), min(h, cor_y + 20)])
                labels.append(human_keypoint_labels[n])
                part_line[n] = (cor_x, cor_y)  # 有效的关键点
            # 关键点之间的连线
            poses[ren_src_index] = part_line
    return labels, boxes, poses


class PosePredictor:
    def __init__(self, model_path, device):
        self.model = onnxruntime.InferenceSession(model_path,
                                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        input_names = self.model.get_inputs()
        output_names = self.model.get_outputs()
        self.output_names = [n.name for n in output_names]
        self.input_name = [n.name for n in input_names][0]
        self.device_type, self.device_id = (device, 0) if device == "cpu" else device.split(":")
        self.device_id = int(self.device_id)
        x = np.zeros((1, 3, 256, 192), np.float32)
        x = onnxruntime.OrtValue.ortvalue_from_numpy(x, device_type=self.device_type, device_id=self.device_id)
        self.model.run(self.output_names, {self.input_name: x})

    @torch.no_grad()
    def predict(self, img, boxes, labels, fn=0):
        ren_boxes = []
        ren_indexes = []
        for index, label in enumerate(labels):
            if label == 'ren':
                ren_boxes.append(boxes[index])
                ren_indexes.append(index)
        if len(ren_boxes) > 0:
            boxes = torch.Tensor(ren_boxes)
            if img is None:
                raise Exception("no image is given")
            if boxes is None or boxes.nelement() == 0:
                return None
            input_size = 256, 192
            inps = []
            cropped_boxes = torch.zeros(boxes.size()[0], 4)
            # 获得画面中的人的区域
            for i, box in enumerate(boxes):
                _img, cropped_box = _transform(img, box)
                inps.append(_img)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
            # hm为预测的关键点
            print("inps", inps[0].shape)
            t0 = time.time()
            x = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(inps, dtype=np.float32),
                                                         device_type=self.device_type, device_id=self.device_id)
            heatmap = self.model.run(self.output_names, {self.input_name: x})[0]
            print(f"{self.device_type}inference time:", time.time() - t0)
            # for i in range(0, 17):
            #     h1 = heatmap[0, i, :, :]
            #     h_img = (h1 * 255).astype(np.uint8)
            #     cv2.imwrite(f"{output_dir}/h{i}_img_resz.png", cv2.resize(h_img, (192, 256)))
            #     cv2.imwrite(f"{output_dir}/h{i}_img.png", h_img)
            print("heatmap shape", heatmap.shape)
            if is_flip:
                heatmap = flip_heatmap(heatmap[int(len(heatmap) / 2):], shift=True)
            heatmap = torch.from_numpy(heatmap)
            keypoints = get_keypoints(ren_indexes, boxes, heatmap, cropped_boxes, fn)
            print(keypoints)
            pose_labels, pose_boxes, poses = get_pose_boxes(img, keypoints)
            return pose_labels, pose_boxes, poses


if __name__ == '__main__':
    import sys, os

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    np.set_printoptions(suppress=True,
                        precision=10,
                        threshold=sys.maxsize,
                        linewidth=150)
    is_flip = False
    weights = r"weights/fastpose_ret50.onnx"
    img: "ndarray" = cv2.imread("images/swote.png")  # 256 192 3
    m = PosePredictor(weights, "cuda:0")
    res = m.predict(img, [[57, 64, 201, 392]], ["ren"])
    # res = res["result"][0]["keypoints"].numpy().astype(np.int)
    # print("len:>>", len(res))
    for pose in res[2].values():
        for i in pose.items():
            print(i)
    if res is not None:
        plot_poses(img, res[2])
    cv2.imwrite("outputs/swote.jpg", img)
    cv2.destroyAllWindows()
