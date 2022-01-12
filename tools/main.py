import argparse
import jittor as jt
from lib.runners import Runner, PMGRunner
from lib.configs import init_cfg 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )

    parser.add_argument(
        "--no_cuda",
        action='store_true'
    )

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )

    parser.add_argument('--pmg', dest='pmg', action='store_true')
    parser.set_defaults(pmg=False)
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1

    assert args.task in ["train","val","test"],f"{args.task} not support, please choose [train,val,test]"
    
    if args.config_file:
        init_cfg(args.config_file)

    runner = PMGRunner() if args.pmg else Runner()
    

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()

if __name__ == '__main__':
    main()