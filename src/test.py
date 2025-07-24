import argparse
import time
from src.utils import check_io, human_time
from src.test_commands import prepare_by_icp, icp


def main():
    parser = argparse.ArgumentParser(
        description="Структурований TEST CLI-інструмент"
    )
    subparsers = parser.add_subparsers(dest="TEST", help="Доступні команди")

    # icp
    parser_icp_p = subparsers.add_parser("icp_file_pipeline", help="Вивести ICP Pipeline")
    parser_icp_p.add_argument("-i", "--input", required=True, help="Вхідний LAS/LAZ файл")
    parser_icp_p.add_argument("-p", "--pipeline", help="JSON pipeline (default pipelines/last_pipeline.json)")
    parser_icp_p.set_defaults(func=prepare_by_icp.prepare_las_file_by_icp)
    parser_icp_p.set_defaults(io_check={"input_keys": ['input'], "output_keys": []})

    # icp
    parser_icp = subparsers.add_parser("icp", help="Вивести ICP")
    parser_icp.add_argument("-i", "--input", required=True, help="Вхідний LAS/LAZ файл")
    parser_icp.set_defaults(func=icp.prepare_las_file_by_icp)
    parser_icp.set_defaults(io_check={"input_keys": ['input'], "output_keys": []})

    args = parser.parse_args()
    # Централізований IO-check
    if hasattr(args, "io_check"):
        check_io(args, **args.io_check)

    if hasattr(args, "func"):
        start = time.time()
        args.func(args)
        end = time.time()
        print(f"\n⏱ Час виконання: {human_time(end - start)}")
    else:
        parser.print_help()
