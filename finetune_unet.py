from accelerate.commands import launch


def main():
    parser = launch.launch_command_parser()
    args = parser.parse_args()
    launch.launch_command(args)
