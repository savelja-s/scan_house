import os
import json
from datetime import datetime

DEFAULT_PIPELINE_FILE = 'pipelines/ICPPrepPipeline.json'


def prepare_las_file_by_icp(args):
    # Визначення файлу пайплайну
    pipeline_path = args.pipeline if args.pipeline and os.path.exists(args.pipeline) else DEFAULT_PIPELINE_FILE

    # Зчитування пайплайну
    with open(pipeline_path, "r") as f:
        pipeline_json = json.load(f)

    # Формування імені вихідного файлу
    input_file_name = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", "prepare_las_file", input_file_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"pre_clasfy_icp_{timestamp}.las")

    # Замінити всі входження {input_file} та {output_file}
    def replace_placeholders(obj):
        if isinstance(obj, dict):
            return {k: replace_placeholders(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_placeholders(x) for x in obj]
        elif isinstance(obj, str):
            return obj.replace("{input_file}", args.input).replace("{output_file}", output_file)
        else:
            return obj

    updated_pipeline = replace_placeholders(pipeline_json)

    print(f"Pipeline: {updated_pipeline}")
    print(f"Output LAS file will be: {output_file}")

    # Далі можна виконувати PDAL Pipeline, якщо це потрібно:
    import pdal
    pipeline = pdal.Pipeline(json.dumps(updated_pipeline))
    pipeline.execute()

    return output_file
