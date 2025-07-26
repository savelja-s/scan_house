#!/usr/bin/env Rscript

# --- Бібліотеки ---
library(lidR)

# --- Аргументи командного рядка ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Використання: Rscript extract_trees.R input.las output.las [algo]\n")
  quit(status = 1)
}
input_file  <- args[1]
output_file <- args[2]
algo_name   <- ifelse(length(args) >= 3, args[3], "dalponte2016")

# --- Читання LAS ---
las <- readLAS(input_file)
if (is.empty(las)) stop("LAS-файл порожній або пошкоджений")

# --- Автоматична класифікація ground (якщо треба) ---
if (sum(las$Classification == 2, na.rm = TRUE) == 0) {
  cat("Ground points not found — running ground classification (PMF)...\n")
  las <- classify_ground(las, pmf(ws = c(3, 6, 12, 24), th = c(0.2, 0.4, 0.8, 1.6)))
  if (sum(las$Classification == 2, na.rm = TRUE) == 0) {
    stop("Не вдалося знайти або класифікувати точки ground (земля).")
  }
}

# --- Нормалізація висоти ---
cat("Normalizing heights (building DTM)...\n")
las_norm <- normalize_height(las, tin())

# --- Canopy Height Model (CHM) ---
cat("Building Canopy Height Model...\n")
chm <- rasterize_canopy(las_norm, res = 0.5, p2r(0.2))

# --- Пошук вершин дерев ---
cat("Locating treetops...\n")
ttops <- locate_trees(chm, lmf(ws = 5, hmin = 2))

# --- Вибір алгоритму сегментації ---
cat(sprintf("Segmenting trees using: %s ...\n", algo_name))
algo <- switch(algo_name,
  "dalponte2016" = dalponte2016(chm, ttops),
  "li2012"       = li2012(),
  "silva2016"    = silva2016(chm, ttops),
  stop(paste("Невідомий алгоритм сегментації:", algo_name))
)

# --- Сегментація дерев ---
las_trees <- segment_trees(las_norm, algo)

# --- Фільтрація тільки точок дерев ---
las_only_trees <- filter_poi(las_trees, !is.na(treeID))

# --- Збереження результату ---
writeLAS(las_only_trees, output_file)
cat("Готово! Файл з точками дерев збережено:", output_file, "\n")
