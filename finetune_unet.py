import argparse
import pathlib

from accelerate.commands import launch


def get_parser():
    "parser for simple argument with sensible defaults"

    parser = argparse.ArgumentParser(
        description="Finetune UNet for Stable Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="/App/data/image-folders/train-1",
        help="Image folder to use for training",
    )
    parser.add_argument(
        "--instance-prompt",
        type=str,
        default="a grayscale microscopic image of human embryo at phase t2",
        help="Instance prompt for SD, change the phase appropriately",
    )
    parser.add_argument(
        "--steps-per-image",
        type=int,
        default=550,
        help="How many steps per image should be performed when training",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="stable-diffusion-2-1-fine-tune-unet-lastBen",
        help="Project name in WandB",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        required=True,
        help="Run name in WandB",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="trail23-medical-image-diffusion",
        help="username or team name where to log the progress"
    )
    parser.add_argument(
        "--save-model-to",
        type=str,
        default="/App/models/stable_diffusion/",
        help="Where the saved model should be saved",
    )
    parser.add_argument(
        "--class-data-dir-parent",
        type=str,
        default="/App/data/class_data_dir",
        help="where the class data dir folder will be created"
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default="/App/models/stable_diffusion/phase-t2-sessions",
        help="Where the checkpoints should be saved",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        help="After how many steps do want the checkpoint to be save",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=200,
        help="After how many steps do you want to want to generate validation images",
    )
    parser.add_argument(
        "--num-validation-images",
        type=int,
        default=20,
        help="How many images do you want to generate for each validation",
    )

    # to train on lower memory
    parser.add_argument(
        "--low-memory-optimizations",
        action="store_true",
        help="Add optimization to allow training on 12GB GPUs",
    )

    # accelerate arguments
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes for accelerate",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Number of machines for accelerate",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
    )
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=["no"] + [b.lower() for b in launch.DYNAMO_BACKENDS],
        help="Choose a backend to optimize your training with dynamo, see more "
        "at https://github.com/pytorch/torchdynamo.",
    )

    return parser


def count_images(image_folder: str):
    "count the number of images in the image folder"
    image_dir = pathlib.Path(image_folder) / "images"
    return len(list(image_dir.iterdir()))


def as_arg_list(args: argparse.Namespace):
    "re-create the arguments as a list to forward them after checking"

    total_steps = count_images(args.image_folder) * args.steps_per_image

    terms: list[str] = [
        "--wandb_project",
        args.wandb_project_name,
        "--Session_dir",
        args.session_dir,
        "--wandb_run_name",
        args.wandb_run_name,
        "--wandb_entity",
        args.wandb_entity,
        "--train_only_unet",
        "--checkpointing_steps",
        str(args.checkpoint_steps),
        "--save_starting_step",
        str(args.checkpoint_steps),
        "--save_n_steps",
        str(args.checkpoint_steps),
        "--class_data_dir",
        args.class_data_dir_parent + "/" + args.wandb_run_name,
        "--pretrained_model_name_or_path",
        "stabilityai/stable-diffusion-2-1-base",
        "--instance_data_dir",
        args.image_folder + "/images",
        "--output_dir",
        args.save_model_to + "/" + args.wandb_run_name,
        "--with_prior_preservation",
        "--prior_loss_weight",
        "1.0",
        "--instance_prompt",
        args.instance_prompt,
        "--class_prompt",
        "a grayscale microscopic image of a human embryo",
        "--resolution",
        "512",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "7e-6",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--num_class_images",
        "200",
        "--max_train_steps",
        str(total_steps),
        "--num_validation_images",
        str(args.num_validation_images),
        "--validation_steps",
        str(args.validation_steps),
        "--validation_prompt",
        args.instance_prompt,
    ]

    if args.low_memory_optimizations:
        terms += [
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--set_grads_to_none",
        ]

    return terms


def main():
    dream_booth_parser = get_parser()
    dream_booth_args = dream_booth_parser.parse_args()

    # add default accelerate args to suppress warnings
    accelerate_args = [
        "--num_processes",
        str(dream_booth_args.num_processes),
        "--num_machines",
        str(dream_booth_args.num_machines),
        "--mixed_precision",
        dream_booth_args.mixed_precision,
        "--dynamo_backend",
        dream_booth_args.dynamo_backend,
        "-m",
        "nsw_da_medical_image.stable_diffusion.train_dreambooth_lastBen",
    ]

    accelerate_parser = launch.launch_command_parser()
    accelerate_cli = accelerate_args + as_arg_list(dream_booth_args)
    args = accelerate_parser.parse_args(accelerate_cli)

    launch.launch_command(args)


if __name__ == "__main__":
    main()
