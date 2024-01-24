# count all images in the dataset
import pathlib
import typing
import pandas as pd

from .enums import FocalPlane, Video
from .dataset import read_phase_ranges, Phase, PhaseRange


class DSRow(typing.TypedDict):
    "rows of the DataFrame"
    focal_plane: int
    video: str
    frame: int


def check_df(image_df: pd.DataFrame):
    planes_u = image_df["focal_plane"].unique()
    video_u = image_df["video"].unique()
    frames_u = image_df["frame"].unique()

    for plane_ in planes_u:
        for video_ in video_u:
            frame_l = image_df[
                (image_df["focal_plane"] == plane_) & (image_df["video"] == video_)
            ]
            if sorted(frame_l) != sorted(frames_u):
                raise ValueError(f"inconsistency for {plane_=!r} and {video_=!r}")

    for video_ in video_u:
        for frame_ in frames_u:
            plane_l = image_df[
                (image_df["video"] == video_) & (image_df["frame"] == frame_)
            ]
            if sorted(plane_l) != sorted(planes_u):
                raise ValueError(f"inconsistency for {video_=!r} and {frame_=!r}")

    for frame_ in frames_u:
        for plane_ in planes_u:
            video_l = image_df[
                (image_df["frame"] == frame_) & (image_df["focal_plane"] == plane_)
            ]
            if sorted(video_l) != sorted(video_u):
                raise ValueError(f"inconsistency for {frame_=!r} and {plane_=!r}")


# TODO annotations


def analyze(
    processed_dataset: pathlib.Path,
    detailed_check: bool,
):
    "see DSRow TypedDict for the DataFrame structure"
    images: list[DSRow] = []

    # load all frames
    prefix = "embryo_dataset"
    for fp_idx, fp in enumerate(FocalPlane):
        dir = processed_dataset / (prefix + fp.suffix)
        if not dir.is_dir():
            raise RuntimeError(f"{dir} not found for focal plane {fp}")
        for video_dir in dir.iterdir():
            video_name = video_dir.name
            if not video_dir.is_dir():
                raise RuntimeError(f"expected {video_dir} to be a directory of images")
            for image in video_dir.iterdir():
                suffix = image.stem.split("_")[-1]
                if not image.is_file():
                    # print(f"unexpected non-file: {image} (skipped)")
                    continue
                if len(suffix) < 4 or suffix[:3] != "RUN":
                    raise RuntimeError(rf"expected {image} to have '_RUN\d+' suffix")
                frame_idx = int(suffix[3:])
                images.append(
                    {"focal_plane": fp_idx, "video": video_name, "frame": frame_idx}
                )

    image_df = pd.DataFrame(images)
    if detailed_check:
        check_df(image_df)

    # Video annotations: where does each phase start and end ?
    annotation_dict: dict[Video, dict[Phase, PhaseRange]] = {}
    for video in Video:
        phases = read_phase_ranges(processed_dataset, video)
        annotation_dict[video] = phases

    # Video timestamps: what does each frame index represent ?
    time_dir = prefix + "_time_elapsed"
    time_dict: dict[Video, dict[int, float]] = {}
    for video in Video:
        time_elapsed = pd.read_csv(
            processed_dataset / time_dir / f"{video.directory}_timeElapsed.csv",
            index_col=None,
            header=0,
        )

        timestamps: dict[int, float] = {}
        for _, idx, time in time_elapsed.itertuples():
            timestamps[idx] = time

        # discard frame #0
        if 0 in timestamps:
            del timestamps[0]

        time_dict[video] = timestamps

    # Grades
    grade_dict: dict[Video, tuple[str, str]] = {}
    grade_df = pd.read_csv(
        processed_dataset / f"{prefix}_grades.csv",
        index_col=None,
        header=0,
        keep_default_na=False,  # don't map 'NA' to 'NaN'
    )
    for _, video, na0, na1 in grade_df.itertuples():
        grade_dict[Video(video)] = (na0, na1)

    return image_df, annotation_dict, time_dict, grade_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("...")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--detailed-check", action="store_true", dest="detailed_check")
    parser.set_defaults(detailed_check=False)
    args = parser.parse_args()
    analyze(pathlib.Path(args.dataset), args.detailed_check)
