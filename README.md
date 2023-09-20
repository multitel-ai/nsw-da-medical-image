# NSW-DA-Medical-Image

Repository for the medical image part of the data augmentation project for the Nantes Summer Workshop (2023).

**A note on the "package" vs directory of script distinction**: because we use a package for all the libraries, we can import sub-modules at any depth without issues. **However**, if we want to run the modules, we cannot execute them as scripts. So the following may not work (it depends on imports in the file) :

```sh
python nsw_da_medical_image/dataset_util/dataset.py ../data/extracted  # will not work
```

But this will work.

```sh
python -m nsw_da_medical_image.dataset_util.dataset ../data/extracted  # ok
```

Which is why it is easier to have all our experiments, and other "runnable" stuff as script, outside of the library :

```sh
.
├── checkpoints
├── ...
├── nsw_da_medical_image  # this is the main package, all the library code go in there
│   ├── __init__.py
│   ├── dataset_util
│   │   ├── __init__.py
│   │   ├── extract_dataset.py
│   │   ├─ ...
│   │   └── util.py
├── ...
├── play_video.py    # this is a script to run
├── experiment_1.py  # this is a script to run
└── yolo.ipynb       # this is a notebook to play with the library
```

## Dataset util(s)

### Directory structure

The current idea for the structure is to have something like this :

```sh
data
├── all-synthetic-runs
│   ├── run1
│   │   ├── img0.jpg
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── metadata.json
│   └── run2
│       ├── img0.jpg
│       ├── img1.jpg
│       ├── img2.jpg
│       └── metadata.json
├── archives
│   ├── embryo_dataset_annotations.tar.gz
│   ├── embryo_dataset_F-15.tar.gz
│   ├── embryo_dataset_F15.tar.gz
│   ├── embryo_dataset_F-30.tar.gz
│   ├── embryo_dataset_F30.tar.gz
│   ├── embryo_dataset_F-45.tar.gz
│   ├── embryo_dataset_F45.tar.gz
│   ├── embryo_dataset_grades.csv
│   ├── embryo_dataset.tar.gz
│   └── embryo_dataset_time_elapsed.tar.gz
├── class_data_dir
│   ├── ...
├── extracted
│   ├── embryo_dataset
│   ├── embryo_dataset_annotations
│   ├── embryo_dataset_F-15
│   ├── embryo_dataset_F15
│   ├── embryo_dataset_F-30
│   ├── embryo_dataset_F30
│   ├── embryo_dataset_F-45
│   ├── embryo_dataset_F45
│   ├── embryo_dataset_grades.csv
│   └── embryo_dataset_time_elapsed
└── image-folders
    └── training-one-per-phase-per-vid
        ├── descriptions
        │   ├── img0.txt
        │   └── img1.txt
        ├── images
        │   ├── img0.jpg
        │   └── img1.jpg
        └── metadata.csv
```

Initially, `data/extracted` should either : (a) be empty, or (b) not exist at all. On the other hand, `data/archives` should have the exact content that is listed above. If one archive is missing, or any other file or directory is found in this directory, an error will be returned before anything is extracted. Likewise, if `data/extracted` exists and is not empty, no archive will be extracted and an error message will be returned.

The directory `all-synthetic-runs` is where synthetic dataset will be after inference. All images in a run have been generated with the same prompt, with the same model, and everything. These information are stored inside the `metadata.json` file, with this structure :

```json
{
    "datetime": "2023-09-11T11:56:51.976986",
    "prompt": "a microscopic image of human embryo at phase t7 recorded at focal plane F+00",
    "phase": "t7",
    "focal-plane": "F+00",
}
```

The directory `images-folders` contains different image folders that can be generated to train stable diffusion easily. The `metadata.csv` file has this structure :

```csv
filename,label,phase
0.jpeg,F+00_BM016-5_146,t2
1.jpeg,F+00_BJ492-8_127,t3
2.jpeg,F+00_PC809-7_106,t2
3.jpeg,F+00_MM84-8_164,t2
4.jpeg,F+00_MM445-2-9_135,t2
5.jpeg,F+00_GS811-3_193,t3
6.jpeg,F+00_OJ319-7_126,t2
7.jpeg,F+00_CE525-2_146,t2
8.jpeg,F+00_LV723-9_105,t2
9.jpeg,F+00_PA214-5_150,t2
```

The directory `/App/data/class_data_dir` is used by Stable Diffusion to generate a few images that will be used during training

### `extract_dataset.py`

How should you extract all the dataset ?

Step 1 : have all archives prepared in a directory. We will not use this directory after that anymore, and the script will not modify anything inside this directory.

```sh
/data/archives
├── embryo_dataset_annotations.tar.gz
├── embryo_dataset_F-15.tar.gz
├── embryo_dataset_F15.tar.gz
├── embryo_dataset_F-30.tar.gz
├── embryo_dataset_F30.tar.gz
├── embryo_dataset_F-45.tar.gz
├── embryo_dataset_F45.tar.gz
├── embryo_dataset_grades.csv
├── embryo_dataset.tar.gz
└── embryo_dataset_time_elapsed.tar.gz
```

Then we run this extraction module to extract all the data :

```sh
python -m nsw_da_medical_image.dataset_util.extract_dataset /data/archives /data/extracted
```

Note that :

- it is not necessary for `/data/extracted` to exist before this call, but **if it exists**, it **must** be empty
- before doing anything, the modules makes sure that all archives are downloaded, if not all of them are there, nothing will be extracted and an error will be returned
- it uses multiple processes so it will be heavy on CPU usage

After a successful run, here is what /data/extract will look like :

```sh
/data/extracted
├── embryo_dataset
├── embryo_dataset_annotations
├── embryo_dataset_F-15
├── embryo_dataset_F15
├── embryo_dataset_F-30
├── embryo_dataset_F30
├── embryo_dataset_F-45
├── embryo_dataset_F45
├── embryo_dataset_grades.csv
└── embryo_dataset_time_elapsed
```

## `make_image_folder.py`

This is how to generate the image folders.

Here is the list of all **required** arguments :

- dataset: this is the source of extracted videos.
- split-file: this is the split.json file.
- set: this is the set to use (must be either train, test, or eval).
- phases: this is the list of phases to use. You may use `all` to specify all phases, all values must be space separated. If `all` is given, no other additional values may be given.
- image-folder-parent: this is where the image folder will be created

And here is the list of **optional** arguments :

- image-folder-name: a name for the image folder (recommended), otherwise a new random name will be used.
- `--shuffle` or `--no-shuffle` to shuffle or not the dataset, the default is to shuffle.
- seed: the seed for the PRG to shuffle the images.

Here are two examples :

```sh
python make_image_folder.py --dataset /App/data/extracted/ --split-file split.json --set eval --phases t6 t7 t8 --image-folder-parent /App/data/image-folders --image-folder-name v6-8-det-s-0 --shuffle --seed 0
```

```sh
python make_image_folder.py --dataset /App/data/extracted/ --split-file split.json --set eval --phases all --image-folder-parent /App/data/image-folders
```

## Stable Diffusion Refactor

The code has been refactored such that the shell script is now at the root directory. Here is an invocation inside the container to train the model :

```sh
cd /App/code && python finetune_unet.py --image-folder /App/data/image-folders/training-all-1vp/ --instance-prompt "grayscale microscopic image of a human embryo at phase t2" --wandb-run-name "unet-t3"
```

You may also add `--wandb-entity "your-user-name"` and `--wandb-project-name "your-project-name"` if you want the run to be saved elsewhere.

To generate a synthetic dataset from the trained model, you may use this command :

```sh
cd /App/code && python make_synthetic_dataset.py --phase "t2" --dataset-name "new-synthetic-t2" --model-path /App/models/stable_diffusion/phase-t2
```
