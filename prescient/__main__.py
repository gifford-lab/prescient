#!/usr/bin/env python
# -*- coding: utf-8 -*-
from prescient.commands import *

def main():
    command_list = [process_data, train_model, simulate_trajectories, perturbation_analysis]
    tool_parser = argparse.ArgumentParser(description='Run a prescient command.')
    command_list_strings = list(map(lambda x: x.__name__[len('prescient.commands.'):], command_list))
    tool_parser.add_argument('command', help='prescient command', choices=command_list_strings)
    tool_parser.add_argument('command_args', help='command arguments', nargs=argparse.REMAINDER)
    prescient_args = tool_parser.parse_args()
    command_name = prescient_args.command
    command_args = prescient_args.command_args
    cmd = command_list[command_list_strings.index(command_name)]
    sys.argv[0] = cmd.__file__
    parser = cmd.create_parser()
    args = parser.parse_args(command_args)
    cmd.main(args)



if __name__ == '__main__':
    args = sys.argv
    # if "--help" in args or len(args) == 1:
    #     print("CVE")
    main()
