from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import zipfile

import cv2
from dotmap import DotMap
from gdown import download_folder, download
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from tinycmax.data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
)


@dataclass
class MvsecSequence:
    root_dir: str
    recording: str
    time_window: float | int | None
    count_window: int | None
    chunk_size: int = 100
    seq_len: int | None = None
    time: tuple[float, float] | tuple[int, int] | None = None  # start, end
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None
    gt: list[str] | None = None
    n_stack: int = 5  # consecutive windows to stack into one [2*n_stack, H, W] sample

    def __post_init__(self):
        # checks
        # TODO: implement count window
        assert not (self.time_window is not None and self.count_window is not None)
        assert self.count_window is None
        assert not (self.augmentations is not None and self.gt)  # no augmentations on gt

        # defaults
        self.root_dir = Path(self.root_dir)
        self.sensor_size = (260, 346)  # height, width

        # make paths
        recording = self.recording[:-1]  # remove trailing number
        paths = {
            "data": self.root_dir / recording / f"{self.recording}_data.hdf5",
            "gt": self.root_dir / recording / f"{self.recording}_gt.hdf5",
            "rect_map_x": self.root_dir / recording / "calib" / f"{recording}_left_x_map.txt",
            "rect_map_y": self.root_dir / recording / "calib" / f"{recording}_left_y_map.txt",
            "calibration": self.root_dir / recording / "calib" / f"camchain-imucam-{recording}.yaml",
        }
        assert all(p.exists() for p in paths.values())

        # open large h5 files only once
        self.fs = dict(
            data=h5py.File(paths["data"], "r"),
            gt=h5py.File(paths["gt"], "r"),
        )

        # forward rectification map
        # distorted -> rectified coords
        # provided map, easier than backward, but gives lines in frames due to nearest neighbor
        rect_map_x = np.loadtxt(paths["rect_map_x"])
        rect_map_y = np.loadtxt(paths["rect_map_y"])
        self.fw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # x_rect, y_rect = rect_map[y, x].T

        # backward rectification/undistortion map
        # rectified/undistorted -> distorted coords
        # more work, but prevents lines in accumulated event frames
        with open(paths["calibration"], "r") as f:
            cam_to_cam = yaml.safe_load(f)
        fx, fy, cx, cy = cam_to_cam["cam0"]["intrinsics"]
        K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_rect = np.array(cam_to_cam["cam0"]["projection_matrix"])[:, :3]
        R_rect = np.array(cam_to_cam["cam0"]["rectification_matrix"])
        dist_coeffs = np.array(cam_to_cam["cam0"]["distortion_coeffs"])
        resolution = cam_to_cam["cam0"]["resolution"]  # xy
        rect_map_x, rect_map_y = cv2.fisheye.initUndistortRectifyMap(
            K_dist, dist_coeffs, R_rect, self.K_rect, resolution, cv2.CV_32F
        )
        self.bw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # needs to be .fisheye!

        # get duration of recording
        # don't get full t because of memory usage
        # convert to us
        self.t0, self.tk = (self.fs["data"]["davis/left/events"][[0, -1], 2] * 1e6).astype(np.int64)
        if self.time is not None:
            t0, tk = self.time
            t0 = t0 + self.t0 if t0 is not None else self.t0
            tk = tk + self.t0 if tk is not None else self.tk
            self.t0, self.tk = t0, tk
        self.rec_duration = self.tk - self.t0

        # slice dataset, pre-compute crop and augmentation
        self.reset()

        # set frame shape
        self.frame_shape = (
            self.crop_corners[2] - self.crop_corners[0],
            self.crop_corners[3] - self.crop_corners[1],
        )

        # each dataset item = n_stack consecutive windows stacked into [2*n_stack, H, W]
        # seq_len still controls how many total windows are cut from each random clip
        # drop_last=True ensures every chunk has exactly n_stack windows; the tail
        # (up to n_stack-1 windows) is discarded rather than producing a short stack
        self.chunk_map = batched(range(len(self.t_start)), self.n_stack, drop_last=True)

    def init_slice(self):
        # start and end time
        if self.seq_len is not None:  # randomly-sliced sequence of seq_len
            start, end = self.t0, max(self.t0, self.tk - (self.seq_len + 1) * self.time_window)  # only full
            # t0/tk are always int64 microseconds (converted in __post_init__)
            t_start = np.random.randint(int(start), max(int(start) + 1, int(end)))
            n_full_windows = self.seq_len
        else:  # full sequence
            t_start, t_end = self.t0, self.tk
            n_full_windows = max(1, int((t_end - t_start) / self.time_window))  # at least 1 window

        # window making
        # use linspace because could be floats (no rounding errors?)
        linspace = np.linspace(t_start, n_full_windows * self.time_window + t_start, n_full_windows + 1)
        self.t_start, self.t_end = linspace[:-1], linspace[1:]
        self.seq_duration = n_full_windows * self.time_window

    def init_crop(self):
        if self.crop:
            if len(self.crop) == 2:  # height, width
                h, w = self.crop
                top = np.random.randint(self.sensor_size[0] - h + 1)  # +1 because exclusive
                left = np.random.randint(self.sensor_size[1] - w + 1)
                self.crop_corners = (top, left, top + h, left + w)
            elif len(self.crop) == 4:  # top, left, bottom, right
                self.crop_corners = self.crop
        else:
            self.crop_corners = (0, 0, *self.sensor_size)

    def init_augmentation(self):
        self.augmentation = []
        if self.augmentations is not None:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentation

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # chunk = n_stack consecutive window indices
        chunk = self.chunk_map[idx]
        n = len(chunk)

        # crop corners are the same for every window in this item
        top, left, bottom, right = self.crop_corners

        # structured dtype for event lists (defined once)
        dtype = np.dtype([("t", np.float64), ("y", np.float32), ("x", np.float32), ("p", np.int8)])

        # collect per-window frames and event lists
        frame_list, lst_list, count_list = [], [], []
        for k, i in enumerate(chunk):
            # convert to indices (from us to s again)
            start = bisect_left(self.fs["data"]["davis/left/events"], self.t_start[i] / 1e6, key=lambda x: x[2])
            end = bisect_left(self.fs["data"]["davis/left/events"], self.t_end[i] / 1e6, key=lambda x: x[2])

            # get events as list
            t = self.fs["data"]["davis/left/events"][start:end, 2]  # float64, s
            y = self.fs["data"]["davis/left/events"][start:end, 1]  # float64
            x = self.fs["data"]["davis/left/events"][start:end, 0]  # float64
            p = self.fs["data"]["davis/left/events"][start:end, 3]  # float64 in {-1, 1}

            # rectify list: forward rectification
            if self.rectify:
                x_rect, y_rect = self.fw_rect_map[y.astype(np.int64), x.astype(np.int64)].T
            else:
                x_rect, y_rect = x, y

            # list of events to structured array
            lst = np.empty(len(t), dtype=dtype)
            lst["t"] = t
            lst["y"] = y_rect
            lst["x"] = x_rect
            lst["p"] = p

            # crop list
            mask = (y_rect >= top) & (y_rect < bottom) & (x_rect >= left) & (x_rect < right)
            lst = lst[mask]
            lst["y"] -= top
            lst["x"] -= left

            # make into event count frame
            # use unrectified coordinates, convert p to {0, 1}
            y_t = torch.from_numpy(y.astype(np.int64))
            x_t = torch.from_numpy(x.astype(np.int64))
            p_t = torch.from_numpy(((p + 1) // 2).astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)
            frame.index_put_((p_t, y_t, x_t), torch.ones_like(p_t), accumulate=True)

            # rectify frame: backward rectification
            if self.rectify:
                frame = cv2.remap(
                    frame.numpy().transpose(1, 2, 0), self.bw_rect_map, None, interpolation=cv2.INTER_NEAREST
                )
                frame = torch.from_numpy(frame.transpose(2, 0, 1))

            # crop frame
            frame = frame[..., top:bottom, left:right]

            # discard if few events or same timestamp
            if len(lst) < 10 or lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=dtype)
                frame = torch.zeros_like(frame)

            # normalize time within window [0, 1], then place in global slot [k/n, (k+1)/n]
            if len(lst):
                lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0])
                lst["t"] = (lst["t"] + k) / n

            frame_list.append(frame)
            lst_list.append(lst)
            count_list.append(len(lst))

        # stack n 2-channel frames along channel dim → [2*n, H, W]
        frames = torch.cat(frame_list, dim=0).float()

        # concatenate events from all n windows; timestamps are globally in [0, 1]
        non_empty = [l for l in lst_list if len(l) > 0]
        all_events = np.concatenate(non_empty) if non_empty else np.array([], dtype=dtype)
        total_count = sum(count_list)
        events_np = rfn.structured_to_unstructured(all_events, dtype=np.float32) if len(all_events) else np.zeros((0, 4), np.float32)
        events_tensor = torch.tensor(events_np)  # [total_count, 4]
        counts_tensor = torch.tensor(total_count)

        # targets — use the last window's flow ground truth
        targets = DotMap(_dynamic=False)
        if self.gt and "flow" in self.gt:
            i = chunk[-1]
            # flow_dist_ts is in seconds; t_start/t_end are in microseconds
            gt_start = bisect_left(self.fs["gt"]["davis/left/flow_dist_ts"], self.t_start[i] / 1e6)
            gt_end = bisect_left(self.fs["gt"]["davis/left/flow_dist_ts"], self.t_end[i] / 1e6)
            if gt_start >= gt_end:
                gt_end = gt_start + 1
            gt_flow_id = gt_start + 1
            gt_flow = self.fs["gt"]["davis/left/flow_dist"][gt_start:gt_end]  # (n, 2, H, W)
            gt_flow = gt_flow[..., top:bottom, left:right]
            gt_flow[np.isnan(gt_flow)] = 0
            targets.flow = torch.from_numpy(gt_flow.astype(np.float32))
            targets.flow_id = torch.tensor(gt_flow_id)

        # apply augmentations
        fh, fw = frames.shape[1], frames.shape[2]
        if "flip_t" in self.augmentation:
            # reverse order of the n stacked 2-channel pairs
            frames = frames.reshape(n, 2, fh, fw).flip(0).reshape(2 * n, fh, fw)
            if events_tensor.shape[0] > 0:
                events_tensor[:, 0] = 1.0 - events_tensor[:, 0]
        if "flip_pol" in self.augmentation:
            # swap positive/negative channels within each pair
            frames = frames.reshape(n, 2, fh, fw).flip(1).reshape(2 * n, fh, fw)
            if events_tensor.shape[0] > 0:
                events_tensor[:, 3] *= -1
        if "flip_ud" in self.augmentation:
            frames = frames.flip(1)
            if events_tensor.shape[0] > 0:
                events_tensor[:, 1] = (bottom - top - 1) - events_tensor[:, 1]
        if "flip_lr" in self.augmentation:
            frames = frames.flip(2)
            if events_tensor.shape[0] > 0:
                events_tensor[:, 2] = (right - left - 2) - events_tensor[:, 2]

        # adapt camera matrices to crop and augmentations
        K_rect = self.K_rect.copy()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "vertical" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "horizontal" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = np.linalg.inv(K_rect)
        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        return DotMap(
            frames=frames,
            auxs=DotMap(events=events_tensor, counts=counts_tensor, _dynamic=False),
            targets=targets,
            recording=self.recording,
            K_rect=K_rect,
            inv_K_rect=inv_K_rect,
            _dynamic=False,
        )


def _feedforward_collate(batch):
    """Collate for feedforward model.

    Input:  list of B DotMaps, each with frames [2*N, H, W] and events [E_i, 4].
    Output: DotMap with frames [B, 2*N, H, W] and events padded to [B, E_max, 4].
    """
    frames = torch.stack([s.frames for s in batch])  # [B, 2*N, H, W]

    # pad variable-length event lists to the longest in the batch
    events_list = [s.auxs.events for s in batch]
    max_n = max(e.shape[0] for e in events_list)
    if max_n > 0:
        events = pad_sequence(events_list, batch_first=True)  # [B, E_max, 4]
    else:
        events = torch.zeros(len(batch), 0, 4)
    counts = torch.stack([s.auxs.counts for s in batch])  # [B]

    targets = DotMap(_dynamic=False)
    for k in list(batch[0].targets.keys()):
        targets[k] = torch.stack([s.targets[k] for s in batch])

    return DotMap(
        frames=frames,
        auxs=DotMap(events=events, counts=counts, _dynamic=False),
        targets=targets,
        recording=[s.recording for s in batch],
        K_rect=torch.stack([s.K_rect for s in batch]),
        inv_K_rect=torch.stack([s.inv_K_rect for s in batch]),
        _dynamic=False,
    )


class MvsecDataModule(LightningDataModule):

    gt = ["flow"]

    def __init__(
        self,
        root_dir,
        time_window,
        count_window,
        train_seq_len,
        train_crop,
        train_recordings,
        val_crop,
        val_recordings,
        rectify,
        augmentations,
        batch_size,
        shuffle,
        num_workers,
        download,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.count_window = count_window
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.train_recordings = train_recordings
        self.val_crop = val_crop
        self.val_recordings = val_recordings
        self.rectify = rectify
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.download = download

    def prepare_data(self):
        if self.download:
            # event and gt data in h5
            # parent: https://drive.google.com/drive/folders/1gDy2PwVOu_FPOsEZjojdWEB2ZHmpio8D
            urls = [
                ("indoor_flying", "https://drive.google.com/drive/folders/1CEuvvahWQntNIqXWZhXu_WknsTLm4Sum"),
                ("outdoor_day", "https://drive.google.com/drive/folders/1WUapfrd2DNQNuxPt9IqUHCcPCPKLiNvT"),
            ]

            # download if not there
            for name, url in urls:
                (self.root_dir / name).mkdir(exist_ok=True, parents=True)
                files = download_folder(url, output=str(self.root_dir / name), skip_download=True)
                for f in files:
                    id, _, local_path = f
                    if not Path(local_path).exists():
                        try:
                            download(id=id, output=local_path)
                        except Exception as e:
                            print(e)
                    if ".zip" in local_path:
                        if not (self.root_dir / name / "calib").exists():
                            with zipfile.ZipFile(local_path, "r") as f:
                                f.extractall(self.root_dir / name / "calib")
                        (self.root_dir / name / f"{name}_calib.zip").unlink()

        # all recordings
        # recordings = [
        #     ("indoor_flying1", None),
        #     ("indoor_flying2", None),
        #     ("indoor_flying3", None),
        #     ("indoor_flying4", None),
        #     ("outdoor_day1", None),
        #     ("outdoor_day2", None),
        # ]

        # train on outdoor_day2, validate on part of outdoor_day1 (default)
        train_recordings = (
            [("outdoor_day2", None)] if self.train_recordings is None else self.train_recordings
        )  # override
        val_recordings = [("outdoor_day1", (222.4, 240.4))] if self.val_recordings is None else self.val_recordings

        # store for building datasets later
        self.train_recordings = train_recordings
        self.val_recordings = val_recordings

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
            )
            train_recordings = []
            for rec in self.train_recordings:
                if isinstance(rec, str):
                    rec = (rec, None)
                r, t = rec
                seq = train_sequence(recording=r, time=t)
                train_recordings.extend([rec] * int(seq.rec_duration / seq.seq_duration))
            self.train_dataset = ConcatDataset([train_sequence(recording=r, time=t) for r, t in train_recordings])
            n = self.train_dataset.datasets[0].n_stack
            self.train_frame_shape = (self.batch_size, 2 * n, *self.train_dataset.datasets[0].frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_window=self.count_window,
                crop=self.val_crop,
                rectify=True,
                gt=self.gt,
            )
            for i, rec in enumerate(self.val_recordings):
                if isinstance(rec, str):
                    rec = (rec, None)
                self.val_recordings[i] = rec
            self.val_dataset = ConcatDataset([val_sequence(recording=r, time=t) for r, t in self.val_recordings])
            n = self.val_dataset.datasets[0].n_stack
            self.val_frame_shape = (1, 2 * n, *self.val_dataset.datasets[0].frame_shape)

        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        return InfiniteDataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=_feedforward_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers // 2,
            collate_fn=_feedforward_collate,
        )


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    # get config
    with initialize(config_path="../config/datamodule", version_base=None):
        config = compose(config_name="mvsec", overrides=["download=true"])

    # download data
    datamodule = instantiate(config)
    datamodule.prepare_data()
