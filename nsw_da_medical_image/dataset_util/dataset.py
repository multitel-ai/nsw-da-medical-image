import bisect
import itertools
import operator
import pathlib
import PIL.Image as Image
import typing
import pandas as pd

import torch
import torchvision.transforms as transforms
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


class VideoPhases(typing.NamedTuple):
    # for each phase : first, last
    content: dict[Phase, PhaseRange]

    @staticmethod
    def read(base_path: pathlib.Path, video: str | Video):
        if isinstance(video, Video):
            video = video.directory
        csv_p = base_path / (PREFIX + "_annotations") / f"{video}_phases.csv"
        df = pd.read_csv(csv_p, index_col=None, header=None)
        df.columns = ["phase", "first", "last"]

        _content: dict[Phase, PhaseRange] = {}
        for _, phase, first, last in df.itertuples():
            _content[Phase(phase)] = PhaseRange(first, last)

        return VideoPhases(_content)

    def count_phases(self):
        "return the *reported* number of annotated frames for each phase, actual may be lower"
        return {p: r.count for p, r in self.content.items()}

    def filter_frames(self, frames: typing.Iterable[int]):
        min_frame = min(pr.first for pr in self.content.values())
        max_frame = max(pr.last for pr in self.content.values())

        return [f for f in frames if min_frame <= f and f <= max_frame]

    def annotate(self, frame: int):
        "annotate a frame, it must be annotated in the CSV"
        for phase, range in self.content.items():
            if range.first > frame:
                continue
            if range.last < frame:
                continue
            return phase
        raise ValueError(f"{frame=} is not annotated")

    def annotate_lst(self, frames: typing.Iterable[int]):
        "annotate all frames, they must be annotated in the CSV"
        return [self.annotate(f) for f in frames]


class DataItem(typing.NamedTuple):
    "how the tuples of the Dataset are structured"

    image: torch.Tensor
    phase: int  # index in enums.Phase
    plane: int  # index in enums.FocalPlane
    video: int  # index in enums.Video
    frame_number: int  # RUN(\d{1,3})


class VideoMetadata(typing.NamedTuple):
    video: int  # index in enums.Video
    frames: list[int]  # values in the filenames
    prefix: str  # filename prefix to find the path of the image
    cum_num_frames: int  # amount of frames of all previous videos in the list
    phases: VideoPhases


def _get_video_frames(
    plane_path: pathlib.Path,
    videos: list[Video],
) -> list[VideoMetadata]:
    videos_metadata: list[VideoMetadata] = []
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

        phases = VideoPhases.read(plane_path.parent, _video)
        frame_lst = phases.filter_frames(frame_lst)

        prefix = frame_file.stem[:idx]  # type:ignore
        metadata = VideoMetadata(
            video=_video.idx(),
            frames=sorted(frame_lst),
            prefix=prefix,
            cum_num_frames=_running_count,
            phases=phases
        )
        videos_metadata.append(metadata)
        _running_count += len(frame_lst)

    return videos_metadata


class NSWDataset(Dataset[DataItem]):
    "return: torch.Tensor for the image"

    def __init__(
        self,
        base_path: pathlib.Path,
        videos: list[Video] | None = None,
        planes: list[FocalPlane] | None = None,
        transform: typing.Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        if videos is None:
            videos = list(Video)
        if planes is None:
            planes = list(FocalPlane)

        self.videos = videos
        self.planes = planes
        self.base_path = base_path
        self.transform = transform or transforms.ToTensor()

        self.videos_metadata = _get_video_frames(
            self.base_path / (PREFIX + self.planes[0].suffix),
            self.videos,
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

        return plane, video, frame, metadata.prefix, metadata.phases

    def get_directory(self, plane: FocalPlane, video: Video):
        return self.base_path / (PREFIX + plane.suffix) / video.directory

    def find_image(self, vid_dir: pathlib.Path, prefix: str, frame: int):
        return vid_dir / (prefix + str(frame) + ".jpeg")

    def __getitem__(self, index) -> DataItem:
        plane, video, frame, prefix, phases = self.un_flatten_idx(index)

        image_path = self.find_image(self.get_directory(plane, video), prefix, frame)
        image = Image.open(image_path)

        phase = phases.annotate(frame)

        data = self.transform(image)

        return DataItem(
            image=data,
            phase=phase.idx(),
            plane=plane.idx(),
            video=video.idx(),
            frame_number=frame,
        )

    def all_phases(self) -> list[Phase]:
        "a list of all phases of this dataset with the same order as the items"
        def _annotate(vid_metadata: VideoMetadata):
            return vid_metadata.phases.annotate_lst(vid_metadata.frames)
        return list(itertools.chain.from_iterable(map(_annotate, self.videos_metadata)))


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
        batch_size, = planes.shape
    except ValueError as e:
        raise ValueError("expected single dimension tensors") from e

    labels: list[str] = []
    for idx in range(batch_size):
        labels.append(__label(int(planes[idx]), int(videos[idx]), int(frames[idx])))

    return labels
