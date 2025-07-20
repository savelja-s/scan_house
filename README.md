# Проєкт: Виділення будівель з LAS-хмар

Цей репозиторій містить набір скриптів для поетапної обробки великих LAS/LAZ файлів (2–100 ГБ) з метою автоматичного виділення полігонів будівель та експорту їх у GeoJSON.

## Структура

```
project_root/
├── lidar_files/              # Вхідні LAS/LAZ файли
│   └── 1.las
├── split_las_tiles.py        # Розбиття великого LAS на тайли з перекриттям
├── classification_and_label.py # Модульна класифікація тайлів: ground → vegetation → label
├── cluster_tiles.py          # Кластеризація точок Class=6 (будівлі)
├── merge_clusters.py         # Злиття кластерів у полігони будівель
├── outputs/                  # Папка з результатами (після запуску)
│   └── 1/
│       ├── tiles/            # Тайли, утворені split_las_tiles.py
│       │   ├── tile_0_0.laz
│       │   └── ...
│       ├── classification/   # Результати classification_and_label.py
│       │   ├── ground_removed/
│       │   ├── vegetation_removed/
│       │   └── labeled/
│       ├── clusters/         # Результати cluster_tiles.py
│       │   └── tile_0_0_cluster.laz
│       └── buildings_merged.geojson  # Остаточний GeoJSON будівель
└── README.md                 # Цей файл
```

## Вимоги

* Python ≥ 3.8
* PDAL і PDAL‑Python bindings (`pip install pdal`)
* laspy (`pip install laspy`)
* numpy, geopandas, shapely (`pip install numpy geopandas shapely`)

## План та чекліст

1. **Розбиття файлу на тайли**

   ```bash
   bash split_las_tiles.sh lidar_files/1.las \
     --tile-size 100 \
     --overlap 10 
   ```

   * [x] Перевірити тайли у `outputs/1/`

2. **Класифікація та маркування**

   ```bash
   python classification_and_label.py outputs/1 \
     --workers 8 \
     --scalar 1.25 --slope 0.15 --threshold 0.5 --window 16 \
     --pmf-max 33 --pmf-slope 1.0 --pmf-initial 0.15 \
     --hag-thresh 2.0
   ```

   * [x] Переглянути файли:

     * `outputs/1/classification/ground_removed/`
     * `outputs/1/classification/vegetation_removed/`
     * `outputs/1/classification/labeled/`

3. **Кластеризація точок будівель**

   ```bash
   python cluster_tiles.py outputs/1/classification/labeled \
     --workers 8 \
     --min-points 100 --tolerance 2.0
   ```

   * [x] Перевірити результат у `outputs/1/clusters/`

4. **Злиття кластерів у полігони**

   ```bash
   python merge_clusters.py outputs/1/clusters \
     --buffer 1.0 \
     --output outputs/1/buildings_merged.geojson
   ```

   * [ ] Переглянути `buildings_merged.geojson`

5. **Аналіз та візуалізація**

   * Відкрийте GeoJSON у QGIS чи іншому ГІС.
   * Перевірте точність та висоти будівель.
    