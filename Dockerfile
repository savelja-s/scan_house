FROM continuumio/miniconda3

# Створюємо середовище з PDAL і Python
RUN conda create -n lidar python=3.10 pdal python-pdal -c conda-forge -y

# Активуємо середовище автоматично
SHELL ["conda", "run", "-n", "lidar", "/bin/bash", "-c"]

WORKDIR /app

COPY . /app

# (опційно) Встановити інші залежності
# RUN pip install -r requirements.txt

# CMD ["conda", "run", "--no-capture-output", "-n", "lidar", "python", "main.py"]
