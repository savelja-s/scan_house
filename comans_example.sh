


python run.py hello --name Сергій
python run.py prepare_las -i lidar_files/1.las -o data/train_trees_5x.csv --decimation 5 -p pipelines/last_pipeline.json
python run.py assign_labels -l ml_files/features_1_tail_0_3_1p.las -i ml_files/features_1_tail_0_3_1p.csv -o data/1_for_train_points_1x_labeled.csv --balance 1.0
python run.py train_model -i data/1_for_train_points_1x_labeled.csv -o data/models/tree_classifier.joblib
python run.py predict_trees -m data/models/tree_classifier_nmap_data.joblib -i outputs/1/tile_0_2.laz -o results/trees/1