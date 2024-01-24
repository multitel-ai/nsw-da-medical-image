import bisect
import operator
import pathlib
import PIL.Image as Image
import typing
import pandas as pd

import torch
from torchvision.transforms.functional import to_tensor  # type:ignore
from torch.utils.data.dataset import Dataset

from .enums import Video, FocalPlane, Phase

# see https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


PREFIX = "embryo_dataset"


class PhaseRange(typing.NamedTuple):
    first: int
    last: int

    @property
    def count(self):
        return self.last - self.first + 1


def read_phase_ranges(base_path: pathlib.Path, video: Video):
    "build a mapping indicating the ranges of each phase in a video"

    csv_p = base_path / (PREFIX + "_annotations") / f"{video.directory}_phases.csv"
    df = pd.read_csv(
        csv_p,
        index_col=None,
        header=None,
        names=["phase", "first", "last"],
        dtype={"phase": str, "first": int, "last": int},
    )

    _content: dict[Phase, PhaseRange] = {}
    for _, phase, first, last in df.itertuples():
        _content[Phase(phase)] = PhaseRange(first, last)

    return _content


def count_phase_ranges(phase_dict: dict[Phase, PhaseRange]):
    return {k: v.count for k, v in phase_dict.items()}


def build_frame_dict(frames: typing.Iterable[int], phase_dict: dict[Phase, PhaseRange]):
    "build a mapping indicating all valid frame of each phase"

    if not phase_dict:
        raise ValueError(f"{phase_dict=!r} may not be empty")

    # sort by phase's first frame
    phases_items = sorted(phase_dict.items(), key=lambda tpl: tpl[1].first)
    phase_start_offset = [pr.first for _, pr in phases_items]

    # group frame in each phase
    frame_dict: dict[Phase, set[int]] = {}
    for frame_ in frames:
        phase_item_idx = bisect.bisect_left(phase_start_offset, frame_)
        phase_item_idx = max(0, phase_item_idx - 1)

        phase, pr = phases_items[phase_item_idx]

        # frames may be before the first or after the last phase
        if frame_ < pr.first:
            continue
        if frame_ > pr.last:
            continue

        frame_dict.setdefault(phase, set()).add(frame_)

    return frame_dict


def filter_median_frames(frame_dict: dict[Phase, set[int]]):
    "filter to only select the median frame of each phase"

    def _select_median(val: set[int]) -> set[int]:
        values = sorted(val)

        mid_idx, odd = divmod(len(values), 2)
        if not odd:
            # arithmetic mean: only work for *dense* range (not guaranteed here)
            med = int(0.5 * (values[mid_idx - 1] + values[mid_idx]))
            # make sure that the median is a valid frame
            if med not in val:
                med = values[mid_idx - 1]  # round left
        else:
            med = values[mid_idx]
        return set([med])

    return {k: _select_median(v) for k, v in frame_dict.items()}


def flatten_frame_dict(frame_dict: dict[Phase, set[int]]):
    def _items(phase: Phase, frames: set[int]):
        value = phase.idx()
        return [(k, value) for k in sorted(frames)]

    items: list[tuple[int, int]] = sum(
        (_items(p, fs) for p, fs in frame_dict.items()), []
    )
    # sort by frame: while each phase is already sorted, the phase may have been out of order
    return sorted(items, key=lambda tpl: tpl[0])


class DataItem(typing.NamedTuple):
    "how the tuples of the Dataset are structured"

    image: torch.Tensor
    phase: int  # index in enums.Phase
    plane: int  # index in enums.FocalPlane
    video: int  # index in enums.Video
    frame_number: int  # RUN(\d{1,3})


class VideoMetadata(typing.NamedTuple):
    video: int  # index in enums.Video
    frames: list[tuple[int, int]]  # (frame number, phase index)
    prefix: str  # filename prefix to find the path of the image
    cum_num_frames: int  # amount of frames of all previous videos in the list


def _get_video_frames(
    plane_path: pathlib.Path,
    videos: list[Video],
    select_phases: list[Phase],
    filter_on_median: bool,
) -> list[VideoMetadata]:
    videos_metadata: list[VideoMetadata] = []
    phase_set = set(select_phases)
    _running_count = 0

    for _video in videos:
        _video_dir = plane_path / _video.directory
        frame_lst: list[int] = []
        for frame_file in list(_video_dir.iterdir()):
            if frame_file.name == "F0":
                continue
            assert frame_file.is_file()
            idx = frame_file.stem.find("RUN", -8) + len("RUN")
            frame_number = int(frame_file.stem[idx:])
            frame_lst.append(frame_number)
        if not frame_lst:
            raise RuntimeError(f"{_video_dir} contains no frame file")

        # read info from the video: what are the start/end of each phase ?
        phases = read_phase_ranges(plane_path.parent, _video)
        if not phases:
            raise RuntimeError(f"unexpected: {phases=!r} at {_video_dir}")

        # apply filtering based on the given phases
        phases = {k: v for k, v in phases.items() if k in phase_set}
        if not phases:
            # in the case of phase filtering, it may happen that some videos do not
            # have any frames for certain phases. This can safely be ignored.
            continue

        # load each frame as a tuple (number, phase) where `number` is the filename suffix
        frame_dict = build_frame_dict(frame_lst, phases)
        if filter_on_median:
            frame_dict = filter_median_frames(frame_dict)
        frames = flatten_frame_dict(frame_dict)

        prefix = frame_file.stem[:idx]  # type:ignore
        metadata = VideoMetadata(
            video=_video.idx(),
            frames=frames,
            prefix=prefix,
            cum_num_frames=_running_count,
        )
        videos_metadata.append(metadata)
        _running_count += len(frames)

    return videos_metadata


class NSWDataset(Dataset[DataItem]):
    "return: torch.Tensor for the image"

    def __init__(
        self,
        base_path: pathlib.Path,
        videos: list[Video] | None = None,
        planes: list[FocalPlane] | None = None,
        phases: list[Phase] | None = None,
        transform: typing.Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        filter_on_median: bool = False,
    ) -> None:
        super().__init__()

        if videos is None:
            videos = list(Video)
        if planes is None:
            planes = list(FocalPlane)
        if phases is None:
            phases = list(Phase)
        if transform is None:
            transform = to_tensor

        if not videos:
            raise ValueError("'videos' may not be empty")
        if not planes:
            raise ValueError("'planes' may not be empty")
        if not phases:
            raise ValueError("'phases' may not be empty")

        self.videos = videos
        self.planes = planes
        self.base_path = base_path
        self.transform = transform

        self.videos_metadata = _get_video_frames(
            self.base_path / (PREFIX + self.planes[0].suffix),
            self.videos,
            select_phases=phases,
            filter_on_median=filter_on_median,
        )

    def frames_per_plane(self) -> int:
        last_metadata = self.videos_metadata[-1]
        return last_metadata.cum_num_frames + len(last_metadata.frames)

    def __len__(self) -> int:
        return len(self.planes) * self.frames_per_plane()

    def flatten_idx(self, plane_list_idx: int, metadata_idx: int, frame_list_idx: int):
        metadata = self.videos_metadata[metadata_idx]
        in_plane_idx = metadata.cum_num_frames + frame_list_idx
        if frame_list_idx >= len(metadata.frames):
            raise IndexError(
                f"{frame_list_idx=} out of bound for {Video.from_idx(metadata.video)}"
            )
        return plane_list_idx * self.frames_per_plane() + in_plane_idx

    def un_flatten_idx(self, flattened: int):
        if flattened < 0 or flattened >= len(self):
            raise IndexError(f"index={flattened} out of bound ({len(self)})")

        plane_list_idx, in_plane_idx = divmod(flattened, self.frames_per_plane())

        # binary search for metadata
        metadata_idx_p1 = bisect.bisect_right(
            self.videos_metadata,
            in_plane_idx,
            key=operator.attrgetter("cum_num_frames"),
        )
        metadata_idx = max(0, metadata_idx_p1 - 1)
        metadata = self.videos_metadata[metadata_idx]

        plane = self.planes[plane_list_idx]
        video = Video.from_idx(metadata.video)
        frame = metadata.frames[in_plane_idx - metadata.cum_num_frames]

        return plane, video, frame, metadata.prefix

    def get_directory(self, plane: FocalPlane, video: Video):
        return self.base_path / (PREFIX + plane.suffix) / video.directory

    def find_image(self, vid_dir: pathlib.Path, prefix: str, frame: int):
        return vid_dir / (prefix + str(frame) + ".jpeg")

    def __getitem__(self, index) -> DataItem:
        plane, video, (frame, phase_idx), prefix = self.un_flatten_idx(index)

        image_path = self.find_image(self.get_directory(plane, video), prefix, frame)
        image = Image.open(image_path)

        data = self.transform(image)

        return DataItem(
            image=data,
            phase=phase_idx,
            plane=plane.idx(),
            video=video.idx(),
            frame_number=frame,
        )


def __label(plane: int, video: int, frame: int):
    plane_val = FocalPlane.from_idx(plane)
    video_val = Video.from_idx(video)
    return f"{plane_val.pretty}_{video_val.directory}_{frame}"


def label_single(data: DataItem):
    return __label(data.plane, data.video, data.frame_number)


def label_batch(planes: torch.Tensor, videos: torch.Tensor, frames: torch.Tensor):
    if planes.shape != videos.shape:
        raise ValueError(f"shape mismatch: {planes.shape=} != {videos.shape=}")
    if videos.shape != frames.shape:
        raise ValueError(f"shape mismatch: {videos.shape=} != {frames.shape=}")

    try:
        (batch_size,) = planes.shape
    except ValueError as e:
        raise ValueError("expected single dimension tensors") from e

    labels: list[str] = []
    for idx in range(batch_size):
        labels.append(__label(int(planes[idx]), int(videos[idx]), int(frames[idx])))

    return labels
