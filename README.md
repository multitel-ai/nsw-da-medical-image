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
        ├── description
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
filename,label
0.jpeg,F+00_DML271-2_51
1.jpeg,F+00_LEG557-3_162
2.jpeg,F+00_LT1112-5_189
3.jpeg,F+00_BF924-3_243
4.jpeg,F+00_DI358-3_14
5.jpeg,F+00_GF667-1-1_184
```

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
