conda create -n lidar_scanner python=3.10 pdal python-pdal -c conda-forge -y
conda activate lidar_scanner
conda update -n base -c defaults conda -y &&
  conda config --add channels conda-forge &&
  conda install -c sirokujira python-pcl -y &&
  conda install -c jithinpr2 gtk3 -y &&
  conda install -y ipython jupyter &&
  conda install conda-forge::shapely -y &&
  conda install conda-forge::geojson -y &&
  conda install pdal python-pdal -c conda-forge -y &&
  conda install -c conda-forge tqdm -y &&
  conda install conda-forge::plotly -y &&
  conda install conda-forge::laspy -y &&
  conda install conda-forge::geopandas -y


# conda env export --no-builds > environment.yml