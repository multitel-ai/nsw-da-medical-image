import collections
import pathlib
import random
import typing
import uuid

import numpy as np
import torch
from PIL import Image

from .dataset import NSWDataset, label_single
from .enums import FocalPlane, Phase, Video


def _prepare_folder(
    image_folder_parent: pathlib.Path,
    image_folder_name: str | None,
    on_exist: typing.Literal["return", "raise"],
):
    if not image_folder_parent.is_dir():
        raise ValueError(f"{image_folder_parent=!r} must exist")

    if image_folder_name is None:
        image_folder = image_folder_parent / str(uuid.uuid4())
    else:
        image_folder = image_folder_parent / image_folder_name
        if image_folder.exists():
            if on_exist == "raise":
                raise ValueError(f"{image_folder=!r} already exist")
            elif on_exist == "return":
                return image_folder
            else:
                raise ValueError(f"{on_exist=!r} has an invalid value")
    image_folder.mkdir()
    return image_folder


def _generate_on_indices(
    indices: list[int],
    dataset: NSWDataset,
    image_folder: pathlib.Path,
):
    _placeholder = torch.empty(())
    next_idx = 0

    def next_path() -> pathlib.Path:
        return image_folder / f"{next_idx}.jpeg"

    def save_image(img: Image.Image) -> torch.Tensor:
        nonlocal next_idx

        img.save(next_path())
        next_idx = next_idx + 1

        return _placeholder  # just here for the correct typing

    dataset.transform = save_image

    with open(image_folder / "metadata.csv", "w", encoding="utf8") as metadata_csv:
        metadata_csv.write("filename,label\n")
        for idx in indices:
            img_path = next_path()
            txt_path = img_path.with_suffix(".txt")

            data_item = dataset[idx]  # transform saves the copy
            metadata_csv.write(f"{img_path.name},{label_single(data_item)}\n")

            phase_idx = data_item.phase
            phase_label = Phase.from_idx(phase_idx).label
            plane_label = FocalPlane.from_idx(data_item.plane).pretty

            with open(txt_path, "w", encoding="utf8") as txt_file:
                txt_file.write(f"a microscopic image of human embryo at phase {phase_label} recorded at focal plane {plane_label}\n")

    return image_folder


def make_image_folder_every_phase_vid(
    extracted_path: pathlib.Path,
    image_folder_parent: pathlib.Path,
    image_folder_name: str | None,
    videos: list[Video],
    focal_planes: list[FocalPlane],
    on_exist: typing.Literal["return", "raise"] = "raise",
    shuffle: bool = True,
    seed: int = 0,
):
    ""
    image_folder = _prepare_folder(
        image_folder_parent,
        image_folder_name,
        on_exist,
    )

    class _Item(typing.NamedTuple):
        metadata_idx: int
        frame_list_idx: int

    dataset = NSWDataset(extracted_path, videos, focal_planes)

    selected: list[_Item] = []
    video_metadata_lst = dataset.videos_metadata
    for idx, video_metadata in enumerate(video_metadata_lst):
        frames: dict[Phase, list[int]] = collections.defaultdict(list)
        phases_in_vid = video_metadata.phases.annotate_lst(video_metadata.frames)
        for frame_idx, (frame, phase) in enumerate(
            zip(video_metadata.frames, phases_in_vid)
        ):
            frames[phase].append(frame_idx)

        # select the median frame for each phase
        for phase, lst in frames.items():
            median_frame = int(np.median(lst))
            selected.append(_Item(idx, median_frame))

    prg = random.Random(seed)
    random_planes = prg.choices(focal_planes or list(FocalPlane), k=len(selected))
    plane_idx = 0

    selected_indices: list[int] = []
    for item in selected:
        plane = random_planes[plane_idx]
        plane_idx += 1

        # compute flat idx
        flat_idx = dataset.flatten_idx(
            dataset.planes.index(plane),
            item.metadata_idx,
            item.frame_list_idx,
        )
        selected_indices.append(flat_idx)

    if shuffle:
        prg.shuffle(selected_indices)

    return _generate_on_indices(selected_indices, dataset, image_folder)


def make_balanced_image_folder(
    frames_per_phase: int,
    extracted_path: pathlib.Path,
    image_folder_parent: pathlib.Path,
    image_folder_name: str | None,
    videos: list[Video],
    focal_planes: list[FocalPlane] | None = None,
    on_exist: typing.Literal["return", "raise"] = "raise",
    if_missing: typing.Literal["drop", "raise"] = "raise",
    seed: int = 0,
):
    image_folder = _prepare_folder(
        image_folder_parent,
        image_folder_name,
        on_exist,
    )

    class _Item(typing.NamedTuple):
        metadata_idx: int
        frame_list_idx: int

    frames: dict[Phase, list[_Item]] = collections.defaultdict(list)

    dataset = NSWDataset(extracted_path, videos, focal_planes)

    video_metadata_lst = dataset.videos_metadata
    for idx, video_metadata in enumerate(video_metadata_lst):
        phases_in_vid = video_metadata.phases.annotate_lst(video_metadata.frames)
        for frame_idx, (frame, phase) in enumerate(
            zip(video_metadata.frames, phases_in_vid)
        ):
            frames[phase].append(_Item(idx, frame_idx))

    prg = random.Random(seed)
    random_planes = prg.choices(
        focal_planes or list(FocalPlane), k=len(Phase) * frames_per_phase
    )
    plane_idx = 0

    random_indices: list[int] = []
    for phase, frame_lst in frames.items():
        if len(frame_lst) < frames_per_phase:
            if if_missing == "raise":
                raise ValueError(f"not enough frame for {phase=}: {len(frame_lst)=}")
            elif if_missing == "drop":
                frames_per_phase = len(frame_lst)
            else:
                raise ValueError(f"illegal value for {if_missing=!r}")

        random_frames = prg.choices(frame_lst, k=frames_per_phase)
        for item in random_frames:
            plane = random_planes[plane_idx]
            plane_idx += 1

            # compute flat idx
            flat_idx = dataset.flatten_idx(
                dataset.planes.index(plane),
                item.metadata_idx,
                item.frame_list_idx,
            )
            random_indices.append(flat_idx)

    return _generate_on_indices(random_indices, dataset, image_folder)


def make_image_folder(
    limit_n_images: int,
    extracted_path: pathlib.Path,
    image_folder_parent: pathlib.Path,
    image_folder_name: str | None,
    videos: list[Video] | None,
    focal_planes: list[FocalPlane] | None = None,
    on_exist: typing.Literal["return", "raise"] = "raise",
    seed: int = 0,
):
    image_folder = _prepare_folder(
        image_folder_parent,
        image_folder_name,
        on_exist,
    )

    dataset = NSWDataset(extracted_path, videos, focal_planes)

    # take 'limit_n_images' at random from the dataset
    indices = list(range(len(dataset)))
    prg = random.Random(seed)
    prg.shuffle(indices)
    indices = indices[:limit_n_images]

    return _generate_on_indices(indices, dataset, image_folder)
