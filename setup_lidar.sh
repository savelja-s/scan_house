#!/bin/bash

#conda create -n lidar python=3.10 pdal python-pdal -c conda-forge -y
#conda activate lidar
conda update -n base -c defaults conda -y && \
conda config --add channels conda-forge && \
conda install -c sirokujira python-pcl -y && \
conda install -c jithinpr2 gtk3 -y && \
conda install -y ipython jupyter && \
conda install conda-forge::shapely -y && \
#conda install conda-forge::geojson -y && \
conda install pdal python-pdal -c conda-forge -y && \
conda install -c conda-forge tqdm  -y

# conda env create -f environment.yml
# conda activate lidar_project
# conda env update --file environment.yml --prune

# conda env export --no-builds > environment.yml


echo "Налаштування середовища Lidar завершено!"
