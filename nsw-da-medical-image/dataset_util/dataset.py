import bisect
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


class DataItem(typing.NamedTuple):
    "how the tuples of the Dataset are structured"

    image: torch.Tensor
    phase: int  # index in enums.Phase
    plane: int  # index in enums.FocalPlane
    video: int  # index in enums.Video
    time: float
    phase_prog: float


class VideoMetadata(typing.NamedTuple):
    video: int  # index in enums.Video
    frames: list[int]  # values in the filenames
    prefix: str  # filename prefix to find the path of the image
    cum_num_frames: int  # amount of frames of all previous videos in the list


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

        self.videos_metadata: list[VideoMetadata] = []
        _plane = self.planes[0]
        _running_count = 0
        for _video in self.videos:
            _video_dir = self.base_path / (PREFIX + _plane.suffix) / _video.directory
            frame_lst = []
            for frame_file in list(_video_dir.iterdir()):
                if frame_file.name == "F0":
                    continue
                assert frame_file.is_file()
                idx = frame_file.stem.find("RUN", -8) + len("RUN")
                frame_number = int(frame_file.stem[idx:])
                frame_lst.append(frame_number)
            prefix = frame_file.stem[:idx]  # type:ignore
            metadata = VideoMetadata(
                video=_video.idx(),
                frames=sorted(frame_lst),
                prefix=prefix,
                cum_num_frames=_running_count,
            )
            self.videos_metadata.append(metadata)
            _running_count += len(frame_lst)

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

    def get_phase(self, video: Video, frame: int) -> tuple[int, float]:
        path = (
            self.base_path / (PREFIX + "_annotations") / f"{video.directory}_phases.csv"
        )
        df = pd.read_csv(path, index_col=None, header=None)
        for _, phase, ps, pe in df.itertuples():
            if frame < ps or frame > pe:
                continue
            prog = 1.0
            if pe != ps:
                prog = (frame - ps) / (pe - ps)
            return Phase(phase).idx(), prog
        else:
            min_start = df[df.columns[1]].min()
            if frame < min_start:
                return Phase.beginning.idx(), frame / min_start
            else:
                return Phase.trailing.idx(), 1.0

    def get_time(self, video: Video, frame: int) -> float:
        path = (
            self.base_path
            / (PREFIX + "_time_elapsed")
            / f"{video.directory}_timeElapsed.csv"
        )
        df = pd.read_csv(path, index_col=None, header=0)
        try:
            selection = df[df["frame_index"] == frame].time
        except KeyError:
            raise RuntimeError(f"column 'frame_index' absent in {path}")

        match len(selection):
            case 1:
                return float(selection.item())
            case 0:
                return float("nan")  # TODO we can do better
            case _:
                raise RuntimeError(f"multiple time values for {frame=} ({path})")

    def __getitem__(self, index) -> DataItem:
        plane, video, frame, prefix = self.un_flatten_idx(index)

        image_path = self.find_image(self.get_directory(plane, video), prefix, frame)
        image = Image.open(image_path)

        phase, phase_prog = self.get_phase(video, frame)

        data = self.transform(image)

        return DataItem(
            image=data,
            phase=phase,
            plane=plane.idx(),
            video=video.idx(),
            time=self.get_time(video, frame),
            phase_prog=phase_prog,
        )


def __main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="extracted dataset")
    parser.add_argument(
        "--n-parts", type=int, default=1, help="how many parts should be split"
    )
    parser.add_argument(
        "--part-index", type=int, default=0, help="index of the part to compute"
    )
    args = parser.parse_args()

    if args.part_index >= args.n_parts:
        raise ValueError(f"{args.part_index=} out of bound ({args.n_parts=})")

    videos: list[Video] | None = None
    if args.n_parts > 1:
        videos = Video.split(args.n_parts)[args.part_index]

    ds = NSWDataset(pathlib.Path(args.dataset), videos=videos)
    for idx in range(len(ds)):
        plane, video, frame, _ = ds.un_flatten_idx(idx)
        try:
            ds[idx]
        except OSError as e:
            print((video, plane, frame, str(e)))
        except ZeroDivisionError as e:
            print((video, plane, frame, str(e)))


if __name__ == "__main__":
    __main()