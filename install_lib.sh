conda create -n lidar_scanner python=3.10 pdal python-pdal -c conda-forge -y
conda activate lidar_scanner
conda update -n base -c defaults conda -y &&
  conda config --add channels conda-forge &&
  conda install -c sirokujira python-pcl -y &&
  conda install -c jithinpr2 gtk3 -y &&
  conda install -y ipython &&
  conda install conda-forge::shapely -y &&
  conda install conda-forge::geojson -y &&
  conda install pdal python-pdal -c conda-forge -y &&
  conda install -c conda-forge tqdm -y &&
  conda install conda-forge::plotly -y &&
  conda install conda-forge::laspy -y &&
  conda install conda-forge::geopandas -y &&
  conda install -c conda-forge jupyterlab -y &&
  conda install conda-forge::entwine -y &&
  conda install -c conda-forge open3d -y

# conda env export --no-builds > environment.yml
# conda env create -f environment.yml
# conda activate lidar_project
# conda env update --file environment.yml --prune
#jupyter lab --ip=0.0.0.0 --port=8888 --no-browser


conda install conda-forge::r-lidr
