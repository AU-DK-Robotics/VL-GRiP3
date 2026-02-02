import open3d as o3d
import numpy as np
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from lib.timer import Timer
from lib.utils import load_obj, natural_key
from datasets.indoor import IndoorDataset
from datasets.kitti import KITTIDataset
from datasets.modelnet import get_train_datasets, get_test_datasets


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None,
                                  sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len,
            sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points, batches_len, features=features,
            sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, classes=labels,
            sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, classes=labels,
            sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        return (torch.from_numpy(s_points), torch.from_numpy(s_len),
                torch.from_numpy(s_features), torch.from_numpy(s_labels))


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports via radius search.
    """
    neighbors = cpp_neighbors.batch_query(
        queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1

    for (src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans,
         matching_inds, src_pcd_raw, tgt_pcd_raw, sample) in list_data:
        batched_points_list += [src_pcd, tgt_pcd]
        batched_features_list += [src_feats, tgt_feats]
        batched_lengths_list += [len(src_pcd), len(tgt_pcd)]

    batched_features = torch.from_numpy(
        np.concatenate(batched_features_list, axis=0))
    batched_points   = torch.from_numpy(
        np.concatenate(batched_points_list, axis=0))
    batched_lengths  = torch.from_numpy(
        np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points       = []
    input_neighbors    = []
    input_pools        = []
    input_upsamples    = []
    input_batches_len  = []

    for block_i, block in enumerate(config.architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Accumulate blocks of the layer
        if 'pool' not in block and 'strided' not in block:
            layer_blocks.append(block)
            if (block_i < len(config.architecture) - 1 and
                'upsample' not in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors
        if layer_blocks:
            if any('deformable' in bl for bl in layer_blocks[:-1]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(
                batched_points, batched_points,
                batched_lengths, batched_lengths,
                r, neighborhood_limits[layer])
        else:
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors
        if 'pool' in block or 'strided' in block:
            dl = 2 * r_normal / config.conv_radius
            pool_p, pool_b = batch_grid_subsampling_kpconv(
                batched_points, batched_lengths, sampleDl=dl)
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            pool_i = batch_neighbors_kpconv(
                pool_p, batched_points, pool_b, batched_lengths,
                r, neighborhood_limits[layer])
            up_i = batch_neighbors_kpconv(
                batched_points, pool_p, batched_lengths, pool_b,
                2 * r, neighborhood_limits[layer])
        else:
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i   = torch.zeros((0, 1), dtype=torch.int64)

        # Update inputs for this layer
        input_points      .append(batched_points.float())
        input_neighbors   .append(conv_i.long())
        input_pools       .append(pool_i.long())
        input_upsamples   .append(up_i.long())
        input_batches_len .append(batched_lengths)

        # Prepare for next layer
        batched_points  = pool_p
        batched_lengths = pool_b
        r_normal       *= 2
        layer          += 1
        layer_blocks    = []

    # Ensure raw clouds are numpy arrays
    if isinstance(src_pcd_raw, torch.Tensor):
        src_pcd_raw = src_pcd_raw.cpu().numpy()
    if isinstance(tgt_pcd_raw, torch.Tensor):
        tgt_pcd_raw = tgt_pcd_raw.cpu().numpy()

    return {
        'points':        input_points,
        'neighbors':     input_neighbors,
        'pools':         input_pools,
        'upsamples':     input_upsamples,
        'features':      batched_features.float(),
        'stack_lengths': input_batches_len,
        'rot':           torch.from_numpy(rot),
        'trans':         torch.from_numpy(trans),
        'correspondences': matching_inds,
        'src_pcd_raw':   torch.from_numpy(src_pcd_raw).float(),
        'tgt_pcd_raw':   torch.from_numpy(tgt_pcd_raw).float(),
        'sample':        sample
    }


def calibrate_neighbors(dataset, config, collate_fn,
                        keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    hist_n = int(np.ceil(4 / 3 * np.pi *
                 (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n),
                             dtype=np.int32)

    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn(
            [dataset[i]], config,
            neighborhood_limits=[hist_n] * config.num_layers)
        counts = [torch.sum(mat < mat.shape[0], dim=1).numpy()
                  for mat in batched_input['neighbors']]
        hists  = [np.bincount(c, minlength=hist_n)[:hist_n]
                  for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum       = np.cumsum(neighb_hists.T, axis=0)
    percentiles  = np.sum(cumsum < (
                   keep_ratio * cumsum[-1, :]), axis=0)
    neighborhood_limits = percentiles
    print('\n')
    return neighborhood_limits


def get_datasets(config):
    if config.dataset == 'indoor':
        info_train      = load_obj(config.train_info)
        info_val        = load_obj(config.val_info)
        info_benchmark  = load_obj(
            f'configs/indoor/{config.benchmark}.pkl')
        train_set       = IndoorDataset(
            info_train, config, data_augmentation=True)
        val_set         = IndoorDataset(
            info_val, config, data_augmentation=False)
        benchmark_set   = IndoorDataset(
            info_benchmark, config, data_augmentation=False)

    elif config.dataset == 'kitti':
        train_set     = KITTIDataset(
            config, 'train', data_augmentation=True)
        val_set       = KITTIDataset(
            config, 'val', data_augmentation=False)
        benchmark_set = KITTIDataset(
            config, 'test', data_augmentation=False)

    elif config.dataset == 'modelnet':
        train_set, val_set = get_train_datasets(config)
        benchmark_set      = get_test_datasets(config)

    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set


def get_dataloader(dataset, batch_size=1, num_workers=4,
                   shuffle=True, neighborhood_limits=None):
    # Hard-coded neighbor limits misurati su training set
    neighborhood_limits = [285, 41, 37, 32]
    print("neighborhood:", neighborhood_limits)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(
            collate_fn_descriptor,
            config=dataset.config,
            neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits


if __name__ == '__main__':
    pass


#IS NOT RE-RUN THE CALIB, STRAIGHT HARD CODED