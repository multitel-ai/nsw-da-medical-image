# Temporary ReadMe to Launch FineTuning Unet of Stable Diffusion

## Create the following directories
1. all-synthetic-runs -> The generated images would be stored here!
2. models -> This is where the pretrained model wil be saved
*These directories are to be created in the same directory as lastBen_unet_fine_tune.sh*

## Launching the finetuning
The finetuning can be be launched using ./lastBen_unet_fine_tune.sh $1 $2 $3 $4 $5 $6 $7 where:
1. $1 = dataset directory path eg: /App/data/extracted/train/images
2. $2 = the instance prompt. *Note to change the phase name*
3. $3 = 220 * number of images in $1
4. $4 = specify wandb_run name -> the current class or current experiment you are running
5. $5 = specify path to save stable diffusion model
6. $6 = specify path to save synthetic images
7. $7 = number of images to be generated

./lastBen_unet_fine_tune.sh /App/data/extracted/train/images 'a grayscale microscopic image of human embryo at phase t2' 3300 'phase-t2' /App/models/stable_diffusion/ /App/data/synthetic-images 3


