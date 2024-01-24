import pathlib
import typing
import uuid

import torch
from PIL import Image

from .dataset import NSWDataset, label_single
from .enums import FocalPlane, Phase, Video


_IMAGES = "images"
_DESCRIPTION = "descriptions"


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
    (image_folder / _IMAGES).mkdir()
    (image_folder / _DESCRIPTION).mkdir()
    return image_folder


def _generate_on_indices(
    indices: typing.Iterable[int],
    dataset: NSWDataset,
    image_folder: pathlib.Path,
):
    _placeholder = torch.empty(())
    next_idx = 0

    def next_path():
        image = image_folder / _IMAGES / f"{next_idx}.jpeg"
        description = image_folder / _DESCRIPTION / f"{next_idx}.txt"
        return image, description

    def save_image(img: Image.Image) -> torch.Tensor:
        nonlocal next_idx

        image_path, _ = next_path()
        img.save(image_path)
        next_idx = next_idx + 1

        return _placeholder  # just here for the correct typing

    dataset.transform = save_image

    def description_line(_phase: str, _plane: str):
        if len(dataset.planes) == 1:
            return f"a microscopic image of a human embryo at phase {_phase}\n"
        return f"a microscopic image of a human embryo at phase {_phase} recorded at focal plane {_plane}\n"

    with open(image_folder / "metadata.csv", "w", encoding="utf8") as metadata_csv:
        metadata_csv.write("filename,label,phase\n")
        for idx in indices:
            img_path, txt_path = next_path()

            data_item = dataset[idx]  # transform saves the copy

            phase_idx = data_item.phase
            phase_label = Phase.from_idx(phase_idx).label
            plane_label = FocalPlane.from_idx(data_item.plane).pretty

            metadata_csv.write(
                f"{img_path.name},{label_single(data_item)},{phase_label}\n"
            )

            with open(txt_path, "w", encoding="utf8") as txt_file:
                txt_file.write(description_line(phase_label, plane_label))

    return image_folder


def make_image_folder_every_phase_vid(
    extracted_path: pathlib.Path,
    image_folder_parent: pathlib.Path,
    image_folder_name: str | None,
    select_phases: list[Phase],
    videos: list[Video],
    focal_planes: list[FocalPlane],
    on_exist: typing.Literal["return", "raise"] = "raise",
):
    image_folder = _prepare_folder(
        image_folder_parent,
        image_folder_name,
        on_exist,
    )

    dataset = NSWDataset(
        extracted_path,
        videos,
        focal_planes,
        select_phases,
        filter_on_median=True,
    )

    return _generate_on_indices(range(len(dataset)), dataset, image_folder)
