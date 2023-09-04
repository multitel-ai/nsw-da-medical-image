"""
Synthetic directory structure :

synthetic-datasets
├── run1
│   ├── img0.jpeg
│   ├── img1.jpeg
│   ├── img2.jpeg
│   └── metadata.json
└── run2
    ├── img0.jpeg
    ├── img1.jpeg
    ├── img2.jpeg
    └── metadata.json

with the metadata.json as
{
    "phase": "tPna",
    "focal-plane": "_F-15",
    "generate-datetime": "2023-09-04T15:30:01.459358"
}
"""

import json
import pathlib
import PIL.Image as Image
import typing

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from .enums import FocalPlane, Phase
from .dataset import DataItem


def _parse_metadata(path: pathlib.Path) -> tuple[FocalPlane, Phase]:
    if not path.exists():
        raise ValueError(f"{path.parent} does not contain a metadata file")
    with open(path, "r", encoding="utf8") as metadata_f:
        metadata = json.load(metadata_f)

    try:
        return metadata["focal-plane"], metadata["phase"]
    except KeyError as e:
        raise ValueError(f"metadata at {path} is invalid") from e


class NSWSyntheticDataset(Dataset[DataItem]):
    "NOTE returns -1 for the video, should be discarded"

    def __init__(
        self,
        base_path: pathlib.Path,
        transform: typing.Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.base_path = base_path
        self.transform = transform or transforms.ToTensor()

        self.images = NSWSyntheticDataset._load_images(self.base_path)

    @staticmethod
    def _load_images(base_path: pathlib.Path):
        images: list[tuple[pathlib.Path, FocalPlane, Phase]] = []

        for run in base_path.iterdir():
            plane, phase = _parse_metadata(run / "metadata.json")

            for image in run.iterdir():
                if image.name == "metadata.json":
                    continue
                if image.is_dir():
                    raise ValueError(f"{run} should only contain metadata and images")
                images.append((image, plane, phase))

        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> DataItem:
        img_path, plane, phase = self.images[index]
        return DataItem(
            image=self.transform(Image.open(img_path)),
            phase=phase.idx(),
            plane=plane.idx(),
            video=-1,  # NOTE this is required for compatibility
        )
