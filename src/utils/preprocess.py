import numpy as np
import h5py
import os

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_data(input_file_path, output_dir_path):
    file = h5py.File(input_file_path, 'r')
    create_dir(output_dir_path)
    
    input_np = np.array(file['Train']['input'])
    target_np = np.array(file['Train']['target'])

    assert input_np.shape[0] == target_np.shape[0]
    
    x_mean = np.mean(
        input_np, 
        axis = tuple(range(input_np.ndim-3)), 
        keepdims = True
    )
    
    x_std = np.std(
        input_np, 
        axis = tuple(range(input_np.ndim-3)), 
        keepdims = True
    )
    
    scaled_input = np.divide(
        input_np - x_mean, 
        x_std, 
        out = np.zeros_like(input_np), 
        where = x_std !=0
    )
    
    scaled_target = np.divide(
        target_np - x_mean, 
        x_std, 
        out = np.zeros_like(target_np), 
        where = x_std != 0
    )

    num_frames = input_np.shape[0]

    for i in range(num_frames):
        print('processing frame ' + str(i))
        frame_dir_path = os.path.join(output_dir_path, f'frame_{i}')
        frame_path = os.path.join(frame_dir_path, 'frame_data.hdf5')
        create_dir(frame_dir_path)

        input1 = scaled_input[i][0]
        input2 = scaled_input[i][1]
        target = scaled_target[i][0]

        frame_hdf5 = h5py.File(frame_path, 'w')
        input1_ds = frame_hdf5.create_dataset('input1', data = input1)
        input2_ds = frame_hdf5.create_dataset('input2', data = input2)
        target_ds = frame_hdf5.create_dataset('target', data = target)
        frame_hdf5.close()


if __name__ == '__main__':
    split_data(
        '/scratch/ns4486/numerical-relativity-interpolation/Proca_fiducial_scaled_cropped.hdf5',
        '/scratch/ns4486/numerical-relativity-interpolation/Proca_fiducial_scaled_cropped_scaled'
    )