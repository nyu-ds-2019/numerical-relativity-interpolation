import numpy as np
import cv2
import os


def generate_video_from_tensor(frame_tensor, save_path):
    # shape of frame_tensor is (num_frames, spatial_x, spatial_y, spatial_z)

    num_frames = frame_tensor.shape[0]

    b_marginal = np.array(frame_tensor[:, :, :, 0])
    b_norm = np.uint8(255 * (b_marginal - b_marginal.min()) / (b_marginal.max() - b_marginal.min()))

    all_array = []

    for i in range(num_frames):
        all_array.append(np.array(Image.fromarray(b_norm[i], 'L').convert('RGB')))

    out = cv2.VideoWriter(
        # '/Users/nikhilvs/repos/nyu/numerical-relativity-interpolation/notebooks/test_new.avi', 
        save_path,
        cv2.VideoWriter_fourcc(*'DIVX'), 
        10, 
        (72, 72)
    )

    for i in range(len(all_array)):
        out.write(all_array[i])
    out.release()

    return