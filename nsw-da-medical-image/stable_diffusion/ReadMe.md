# Temporal ReadMe to Launch FineTuning Unet of Stable Diffusion

## In lastBen_unet_fine_tune.sh
1. change `instance_dir_name` to the directory where all the images are found
2. remember to change the phase name in `instance_prompt`
3. change `max_train_steps` to 220 * number of images in `instance_dir_name`

## In launch_finetuning.sh
1. change name the username 

## Create the following directories
1. generated_images -> The generated images would be stored here!
2. models -> This is where the pretrained model wil be saved

## Once all is done run launch_finetuning.sh