# 人体关键点预测
# 17个关键点，按顺序为:
# 鼻子，左眼，右眼，左耳，右耳，左肩，右肩，左肘，右肘，左手，右手
# 左胯，右胯, 左膝，右膝，左踝，右踝
# 颈部关键点neck不是通过模型预测出的，而是通过左肩和右肩的中心点计算得出
human_keypoint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]

# 关键点之间连线
l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head(鼻子->左眼,鼻子->右眼,左眼->左耳,右眼->右耳)
          (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
          (17, 11), (17, 12),  # Body,17为计算出的颈部关键点
          (11, 13), (12, 14), (13, 15), (14, 16)
          ]

# 点的颜色
p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
           # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck(颈部通过计算补充)

# 线的颜色
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],  # 17 body keypoints
               [20 - 3, 23 - 3], [21 - 3, 24 - 3], [22 - 3, 25 - 3], [26 - 3, 42 - 3], [27 - 3, 41 - 3],
               [28 - 3, 40 - 3],
               [29 - 3, 39 - 3], [30 - 3, 38 - 3],
               [31 - 3, 37 - 3], [32 - 3, 36 - 3], [33 - 3, 35 - 3], [43 - 3, 52 - 3], [44 - 3, 51 - 3],
               [45 - 3, 50 - 3],
               [46 - 3, 49 - 3], [47 - 3, 48 - 3],
               [62 - 3, 71 - 3], [63 - 3, 70 - 3], [64 - 3, 69 - 3], [65 - 3, 68 - 3], [66 - 3, 73 - 3],
               [67 - 3, 72 - 3],
               [57 - 3, 61 - 3], [58 - 3, 60 - 3],
               [74 - 3, 80 - 3], [75 - 3, 79 - 3], [76 - 3, 78 - 3], [87 - 3, 89 - 3], [93 - 3, 91 - 3],
               [86 - 3, 90 - 3],
               [85 - 3, 81 - 3], [84 - 3, 82 - 3],
               [94 - 3, 115 - 3], [95 - 3, 116 - 3], [96 - 3, 117 - 3], [97 - 3, 118 - 3], [98 - 3, 119 - 3],
               [99 - 3, 120 - 3],
               [100 - 3, 121 - 3],
               [101 - 3, 122 - 3], [102 - 3, 123 - 3], [103 - 3, 124 - 3], [104 - 3, 125 - 3], [105 - 3, 126 - 3],
               [106 - 3, 127 - 3], [107 - 3, 128 - 3],
               [108 - 3, 129 - 3], [109 - 3, 130 - 3], [110 - 3, 131 - 3], [111 - 3, 132 - 3], [112 - 3, 133 - 3],
               [113 - 3, 134 - 3], [114 - 3, 135 - 3]]

# parser = argparse.ArgumentParser(description='Single-Image Demo')
# # parser.add_argument('--detector', dest='detector',
# #                     help='detector name', default="yolo")
# parser.add_argument('--cfg', type=str, default='./data/256x192_res50_lr1e-3_1x.yaml',
#                     help='experiment configure file name')
# parser.add_argument('--checkpoint', type=str, default='checkpoint/fast_res50_256x192.pth',
#                     help='checkpoint file name')
# parser.add_argument('--min_box_area', type=int, default=0,
#                     help='min box area to filter out')
# parser.add_argument('--gpus', type=str, dest='gpus', default="0,1",
#                     help='choose which cuda device to use by index and input comma to use multi'
#                     'gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
# parser.add_argument('--pose_flow', dest='pose_flow',
#                     help='track humans in video with PoseFlow', action='store_true', default=False)  # False
# parser.add_argument('--pose_track', dest='pose_track',
#                     help='track humans in video with reid', action='store_true', default=False)  # False
# parser.add_argument('--showbox', default=False, action='store_true',
#                     help='visualize human bbox')
# parser.add_argument('--flip', default=False, action='store_true',
#                     help='enable flip testing')
