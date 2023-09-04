import hashlib
import pathlib
import shutil

from extract_dataset import FocalPlane


def hash(file: pathlib.Path):
    if not file.is_file():
        raise ValueError(f"cannot hash a non-file: {file}")
    data = open(file, "rb").read()
    return hashlib.md5(data).hexdigest()


def compare_dirs(video_dir: pathlib.Path, f0_dir: pathlib.Path):

    left_s = set([p.name for p in video_dir.iterdir() if p.name != "F0"])
    right_s = set([p.name for p in f0_dir.iterdir()])

    if left_s != right_s:
        print(f"comparing {video_dir} to {f0_dir}:")
        print(f"\tonly in left {left_s.difference(right_s)}")

    for img in left_s:
        l_hash = hash(video_dir / img)
        r_hash = hash(f0_dir / img)
        if l_hash != r_hash:
            print(f"hash differ for {video_dir/img} vs {f0_dir/img}:")
            print(f"\t{l_hash}")
            print(f"\t{r_hash}")


def analyze(
    processed_dataset: pathlib.Path,
    *,
    remove: bool,
):

    prefix = "embryo_dataset"
    for fp in FocalPlane:  # F0 is the only one that have those folders
        dir = processed_dataset / (prefix + fp.suffix)
        if not dir.is_dir():
            raise RuntimeError(f"{dir} not found for focal plane {fp}")
        for video_dir in dir.iterdir():
            if not video_dir.is_dir():
                raise RuntimeError(f"expected {video_dir} to be a directory of images")
            f0_dir = video_dir / "F0"
            if f0_dir.is_dir():
                compare_dirs(video_dir, f0_dir)
                if remove:
                    shutil.rmtree(f0_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("...")
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    analyze(pathlib.Path(args.dataset), remove=True)
