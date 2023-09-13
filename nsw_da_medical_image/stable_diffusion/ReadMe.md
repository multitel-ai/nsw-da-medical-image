# Temporary ReadMe to Launch FineTuning Unet of Stable Diffusion

The train_dreambooth.py file is originally from https://github.com/TheLastBen/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

Some modifications have been done to the train_dreambooth.py

## Create the following directories
1. all-synthetic-runs -> The generated images would be stored here!
2. models -> This is where the pretrained model wil be saved

## Launching the finetuning
The finetuning can be be launched using ./lastBen_unet_fine_tune.sh $1 $2 $3 $4 $5 $6 $7 where:
1. $1 = dataset directory path eg: /App/data/extracted/train/images
2. $2 = the instance prompt. *Note to change the phase name*
3. $3 = 550 * number of images in $1
4. $4 = specify wandb_run name -> the current class or current experiment you are running
5. $5 = specify path to save stable diffusion model
6. $6 = specify path to save synthetic images
7. $7 = number of images to be generated after finetuning the model
8. $8 = session dir -> Where .ckpt will be kept
9. $9 = after how many steps do want the checkpoint to be save
10. $10 = after how many steps do you want to want to generate validation images
11. $11 = how many images do you want to generate for each validation

./lastBen_unet_fine_tune.sh /App/data/extracted/phase_t2_images/images 'a grayscale microscopic image of human embryo at phase t2' 3300 'phase-t2' /App/models/stable_diffusion/ /App/data/synthetic-images 50 /App/models/stable_diffusion/phase-t2-sessions 500 200 20


