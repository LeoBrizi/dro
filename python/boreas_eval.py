import numpy as np
import os
import os.path as osp

from pyboreas.utils.odometry import (
    compute_kitti_metrics,
    get_sequence_poses,
    get_sequence_poses_gt,
    get_sequences
)

result_path = './output'
gt_path = '/home/ced/Documents/data/boreas/doppler/rss/test'

# New sequences
sequence_type = {
    'boreas-2024-12-03-10-24': 'Suburbs',
    'boreas-2024-12-03-12-54': 'Suburbs',
    'boreas-2025-01-08-10-59': 'Suburbs',
    'boreas-2025-01-08-11-22': 'Suburbs',
    'boreas-2025-01-08-11-44': 'Suburbs',
    'boreas-2025-01-08-12-28': 'Suburbs',

    
    'boreas-2024-12-03-13-13': 'Highway',
    'boreas-2024-12-03-13-34': 'Highway',
    'boreas-2024-12-10-12-07': 'Highway',
    'boreas-2024-12-10-12-24': 'Highway',
    'boreas-2024-12-10-12-38': 'Highway',
    'boreas-2024-12-10-12-56': 'Highway',

    'boreas-2024-12-04-14-28': 'Tunnel',
    'boreas-2024-12-04-14-34': 'Tunnel',
    'boreas-2024-12-04-14-38': 'Tunnel',
    'boreas-2024-12-04-14-44': 'Tunnel',
    'boreas-2024-12-04-14-50': 'Tunnel',
    'boreas-2024-12-04-14-59': 'Tunnel',

    'boreas-2024-12-04-11-45': 'Skyway',
    'boreas-2024-12-04-11-56': 'Skyway',
    'boreas-2024-12-04-12-08': 'Skyway',
    'boreas-2024-12-04-12-19': 'Skyway',
    'boreas-2024-12-04-12-34': 'Skyway',

    '' : 'Unknown'
}


def main():
    # Get the list of folders in the result path
    folders = [f for f in os.listdir(result_path) if osp.isdir(osp.join(result_path, f))]
    folders = sorted(folders)

    # Store the results per sequence type
    results = {key: {} for key in sequence_type.values()}


    for folder in folders:
        print('Processing folder: ', folder)

        try:
            t_err, r_err = eval_odom(osp.join(result_path, folder, 'odometry_result'), gt_path, radar=True)
        except:
            print('Error in sequence: ', folder)
            continue

        print('Mean translation error: ', t_err)
        print('Mean rotation error: ', r_err)

        # Get the sequence type
        if folder not in sequence_type:
            seq_type = 'Unknown'
        else:
            seq_type = sequence_type[folder]

        results[seq_type][folder] = (t_err, r_err)

    print('')
    print('')
    print('Results for ', result_path, ':')

    # Print the results
    for key in results:
        print('')
        print('--------Sequence type: ', key)
        t_errs = [t_err for t_err, _ in results[key].values()]
        r_errs = [r_err for _, r_err in results[key].values()]
        print('Mean translation error: ', np.mean(t_errs))
        print('Mean rotation error: ', np.mean(r_errs))
        print('Details:')
        for folder in results[key]:
            print('Sequence: ', folder, ' t_err: ', np.round(results[key][folder][0],2), '% r_err: ', np.round(results[key][folder][1],5), 'deg/m')








# Copy paste from pyboreas
def eval_odom(pred="test/demo/pred/3d", gt="test/demo/gt", radar=False):
    # evaluation mode
    dim = 2 if radar else 3

    # parse sequences
    seq = get_sequences(pred, ".txt")
    T_pred, times_pred, seq_lens_pred = get_sequence_poses(pred, seq)

    # get corresponding groundtruth poses
    T_gt, _, seq_lens_gt, crop = get_sequence_poses_gt(gt, seq, dim)

    # compute errors
    t_err, r_err, _ = compute_kitti_metrics(
        T_gt, T_pred, seq_lens_gt, seq_lens_pred, seq, pred, dim, crop
    )

    # print out results
    print("Evaluated sequences: ", seq)
    print("Overall error: ", t_err, " %, ", r_err, " deg/m")

    return t_err, r_err





if __name__ == "__main__":
    main()

