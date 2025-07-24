import json
import pdal
import pandas as pd
import os


def prepare_las(args):
    # 1. Визначаємо шлях до pipeline
    pipeline_path = args.pipeline or os.path.join("pipelines", "last_pipeline.json")

    # 2. Завантажуємо pipeline з файлу або генеруємо дефолтний
    if args.pipeline and os.path.exists(args.pipeline):
        with open(args.pipeline, "r") as f:
            pipeline_json = json.load(f)
    else:
        pipeline_json = [
            {"type": "readers.las", "filename": args.input},
            {"type": "filters.decimation", "step": args.decimation},
            {
                "type": "filters.smrf",
                "scalar": 1.25,
                "slope": 0.15,
                "threshold": 0.5,
                "window": 16.0
            },
            {
                "type": "filters.hag_nn"
            }
            # Можна додати інші фільтри
        ]

    # 3. Оновлюємо filename і decimation, якщо задані явно
    # (це дає можливість "перезаписати" значення з файлу через CLI)
    for stage in pipeline_json:
        if stage.get("type") == "readers.las" and args.input:
            stage["filename"] = args.input
        if stage.get("type") == "filters.decimation" and args.decimation is not None:
            stage["step"] = args.decimation

    # Якщо decimation відсутній у pipeline, але параметр задано — додаємо
    if args.decimation is not None and not any(s.get("type") == "filters.decimation" for s in pipeline_json):
        pipeline_json.append({"type": "filters.decimation", "step": args.decimation})

    # 4. Зберігаємо pipeline для аудиту (опціонально)
    # with open(pipeline_path, "w") as f:
    #     json.dump(pipeline_json, f, indent=2)

    # 5. Виконуємо pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    count = pipeline.execute()
    arr = pipeline.arrays[0]

    # 6. Зберігаємо результат
    df = pd.DataFrame(arr)
    df.to_csv(args.output, index=False)
    print(f"Оброблено {count} точок. Вивід: {args.output}")
