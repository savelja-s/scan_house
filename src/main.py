import argparse
from src.commands import hello, prepare_las, assign_labels, train_model, predict_trees
from src.utils import check_io, human_time
import time


def main():
    parser = argparse.ArgumentParser(
        description="Структурований CLI-інструмент"
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступні команди")

    # hello
    parser_hello = subparsers.add_parser("hello", help="Вивести Hello World")
    parser_hello.add_argument("-n", "--name", default="World", help="Ім'я для привітання")
    parser_hello.set_defaults(func=hello.hello)
    parser_hello.set_defaults(io_check={"input_keys": [], "output_keys": []})

    # prepare_las
    parser_prep = subparsers.add_parser("prepare_las", help="Підготувати LAS для тренування")
    parser_prep.add_argument("-i", "--input", required=True, help="Вхідний LAS/LAZ файл")
    parser_prep.add_argument("-o", "--output", required=True, help="Вихідний CSV файл")
    parser_prep.add_argument("-p", "--pipeline", default=None, help="JSON pipeline файл (опціонально)")
    parser_prep.add_argument("-d", "--decimation", type=int, default=1, help="Крок декімінації (default 1)")
    parser_prep.set_defaults(func=prepare_las.prepare_las)
    parser_prep.set_defaults(io_check={"input_keys": ["input"], "output_keys": ["output"]})

    # parser_label
    parser_label = subparsers.add_parser("assign_labels", help="Додати label до CSV на основі LAS-файлу")
    parser_label.add_argument("-l", "--las", required=True, help="LAS/LAZ-файл з позитивним класом")
    parser_label.add_argument("-i", "--input_csv", required=True, help="CSV-файл з усіма точками")
    parser_label.add_argument("-o", "--output", required=True, help="Вихідний CSV з лейблами")
    parser_label.add_argument("--balance", type=float, default=None,
                              help="Зробити undersampling негативного класу до BALANCE * #positive (наприклад, 1.0 для 1:1)")
    parser_label.set_defaults(func=assign_labels.assign_labels)
    parser_label.set_defaults(io_check={"input_keys": ["las", "input_csv"], "output_keys": ["output"]})

    # train_model
    parser_train = subparsers.add_parser("train_model", help="Навчити модель для класифікації дерев")
    parser_train.add_argument("-i", "--input", required=True, help="CSV для навчання")
    parser_train.add_argument("-o", "--output", required=True, help="Вихідний файл моделі (.joblib/.pkl)")
    parser_train.add_argument(
        "--features", nargs="*", default=None,
        help="Явний список ознак для навчання (через кому або декілька параметрів)"
    )
    parser_train.add_argument(
        "--summary", default=None, help="Файл для збереження логу/summary навчання"
    )
    parser_train.set_defaults(func=train_model.train_model)
    parser_train.set_defaults(io_check={"input_keys": ["input"], "output_keys": ["output"]})

    # parser_pred
    parser_pred = subparsers.add_parser("predict_trees", help="Класифікувати дерева у LAS/LAZ файлах моделлю")
    parser_pred.add_argument("-m", "--model", required=True, help="Файл моделі (.joblib/.pkl)")
    parser_pred.add_argument("-i", "--input", required=True, help="LAS/LAZ-файл або папка")
    parser_pred.add_argument("-o", "--output", required=True, help="Папка для результатів")
    parser_pred.add_argument("-p", "--pipeline", default=None,
                             help="JSON pipeline (default pipelines/last_pipeline.json)")
    parser_pred.add_argument("-t", "--threads", type=int, default=4, help="Кількість потоків")
    parser_pred.set_defaults(func=predict_trees.predict_trees)
    parser_pred.set_defaults(io_check={"input_keys": ["model", "input"], "output_keys": []})

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
