# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

library(tidyverse)
library(writexl)

experiment_folder <- "christian/gan"

files <- list.files(paste0("experiment/2022-06-27-matching_conditions_evaluation/", experiment_folder, "/image"))
images <- substr(files, 1, 20) %>% unique()
evaluation <- data.frame(image = images)

write_xlsx(evaluation, paste0("experiment/2022-06-27-matching_conditions_evaluation/", experiment_folder, "/evaluation.xlsx"))
