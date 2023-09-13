import multiprocessing
import pathlib
import shutil
import tarfile

from .enums import FocalPlane


def get_all_archives():
    prefix = "embryo_dataset"
    ext = ".tar.gz"

    lst = [
        prefix + "_annotations" + ext,
        prefix + "_grades.csv",
        prefix + "_time_elapsed.tar.gz",
    ]

    for plane in FocalPlane:
        lst.append(prefix + plane.suffix + ext)

    return lst


def __prep_fn(path_tpl: tuple[pathlib.Path, pathlib.Path, pathlib.Path]):
    src, dst, file = path_tpl

    if file.suffixes == [".tar", ".gz"]:
        # tar.gz -> extract into dst
        tar_file = tarfile.open(file, mode="r")
        tar_file.extractall(dst)
        tar_file.close()
    else:
        # other file -> copy
        dst_file = dst / file.relative_to(src)
        shutil.copy(file, dst_file)


def extract_dataset(
    source_directory: pathlib.Path,
    destination_directory: pathlib.Path,
    n_proc: int | None = None
):
    """
    - source_directory: directory containing all archive files
    - destination directory: destination for the archive contents, must either not exist of be empty
    """
    # FIXME the checks were removed because we only cared about the F0 and it would take too much time
    # when we're ready for all focal-planes, the commented lines should be restored.

    # checklist = sorted([source_directory / f for f in get_all_archives()])
    all_files = sorted(source_directory.iterdir())

    # if all_files != checklist:
    #    absent = set(checklist).difference(all_files)
    #    extras = set(all_files).difference(checklist)
    #    raise ValueError(
    #        f"invalid source directory ({source_directory}): "
    #        f"missing {absent}, found extras {extras}"
    #    )

    if not destination_directory.exists():
        destination_directory.mkdir()
    elif len(list(destination_directory.iterdir())) > 0:
        raise ValueError(
            f"invalid destination ({destination_directory}): should be empty"
        )

    if n_proc is None:
        n_proc = len(all_files)

    print(f"extracting {all_files} to {destination_directory}")
    args = [(source_directory, destination_directory, path) for path in all_files]
    with multiprocessing.Pool(n_proc) as pool:
        pool.map(__prep_fn, args)
