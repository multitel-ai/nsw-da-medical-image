import collections
import pathlib
import random
import typing

from .enums import Phase, Video
from .dataset import VideoPhases


def random_split(videos: list[Video], weights: typing.Iterable[float], prg: random.Random):
    "randomly split `videos` into parts of length proportional to weights"

    weight_sum = sum(weights, 0.0)
    if abs(weight_sum) < 1e-2:
        raise ValueError("weights should approximately sum to 1.0")

    weights = [w / weight_sum for w in weights]
    cum_weights: list[float] = []
    curr_sum = 0.0
    for w in weights:
        curr_sum += w
        cum_weights.append(curr_sum)
    assert abs(cum_weights[-1] - 1.0) < 1e-2, (
        "cumulative weights should reach 1.0",
        cum_weights,
        weights,
    )

    offsets = [round(w * len(videos)) for w in cum_weights]
    assert offsets[-1] == len(videos), (
        "last offset should include all videos",
        offsets,
        cum_weights,
        weights,
    )

    counts = [offsets[0]]
    for offset in offsets[1:]:
        counts.append(offset - counts[-1])
    assert not any(c == 0 for c in counts), (
        "all partitions should be non-empty",
        counts,
        offsets,
        cum_weights,
        weights,
    )

    # list copy and shuffle
    videos = videos[:]
    prg.shuffle(videos)

    partitions: list[list[Video]] = []
    start = 0
    for end in offsets:
        partitions.append(videos[start:end])
        start = end

    return partitions


def sorted_occurrences(ctr: dict[Phase, list[Video]]):
    "`phase` -> all videos having at least one frame with phase `phase`"
    occ = {p: vids for p, vids in ctr.items()}
    return sorted(occ.items(), key=lambda tpl: len(tpl[1]))


def remove_phase(ctr: dict[Phase, list[Video]], phase: Phase):
    "removes a specific `phase` and its associated videos from a given dictionary"

    vid_to_remove = ctr[phase]

    for phase, lst in list(ctr.items()):
        ctr[phase] = [vid for vid in lst if vid not in vid_to_remove]

    phases_to_remove = [phase for phase, lst in ctr.items() if not lst]
    for phase in phases_to_remove:
        ctr.pop(phase)


def count_phases_in_videos(base_path: pathlib.Path, videos: list[Video] | None = None):
    "count the number of frame of each phase in all videos"

    if videos is None:
        videos = list(Video)

    phase_counter: dict[Video, dict[Phase, int]] = {}
    for video in videos:
        vid_phases = VideoPhases.read(base_path, video)
        phase_counter[video] = vid_phases.count_phases()
    return phase_counter


def count_vid_per_phase(
    phase_counter: dict[Video, dict[Phase, int]]
) -> dict[Phase, list[Video]]:
    "list all videos containing at least a frame of a phase `p` in each entry of the dict"
    per_phase_vid_lst: dict[Phase, list[Video]] = collections.defaultdict(list)
    for vid, counter in phase_counter.items():
        for phase, count in counter.items():
            if count == 0:
                continue
            per_phase_vid_lst[phase].append(vid)
    return per_phase_vid_lst


def fair_random_split(
    per_phase_vid_lst: dict[Phase, list[Video]],
    weights: typing.Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 1234567890,
):
    # make a mutable copy
    to_be_sorted = {key: val[:] for key, val in per_phase_vid_lst.items()}

    sets: list[list[Video]] = [list() for _ in weights]

    prg = random.Random(seed)

    # Greedy approach to even split :
    #   - 1. compute all occurrences
    #   - 1. find the phase which is shown in the fewest videos
    #   - 2. split those videos randomly
    #   - 3. remove these videos from each list of the map
    #   - 3. remove this phase (which is split uniformly)
    #   - repeat

    while to_be_sorted:
        # 1: find the n_vids min
        (phase, vids), *_ = sorted_occurrences(to_be_sorted)

        # 2: split those
        vid_split = random_split(vids, weights, prg)

        # 2: merge sets and vid_split
        for set_, new_ones in zip(sets, vid_split):
            set_.extend(new_ones)

        # 3: remove them
        remove_phase(to_be_sorted, phase)

    return sets


def check_fairness(base_path: pathlib.Path, sets: list[list[Video]]):
    counters = [count_phases_in_videos(base_path, set_) for set_ in sets]

    phase_vals: dict[Phase, list[int]] = {}

    for phase in Phase:
        val_per_phase: list[int] = []
        for ctr in counters:
            val = 0
            for vid, p_ctr in ctr.items():
                val += p_ctr[phase]
            val_per_phase.append(val)
        phase_vals[phase] = val_per_phase

    return phase_vals
