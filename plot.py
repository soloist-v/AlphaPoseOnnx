import cv2
from const import l_pair, p_color, line_color


def plot_pose(img, part_line):
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            for n, (cor_x, cor_y) in [(start_p, start_xy), (end_p, end_xy)]:
                if n < len(p_color):
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)
            if i < len(line_color):
                cv2.line(img, start_xy, end_xy, line_color[i], 2)
            else:
                cv2.line(img, start_xy, end_xy, (255, 255, 255), 1)


def plot_poses(img, poses):
    for index, pose in poses.items():
        plot_pose(img, pose)
