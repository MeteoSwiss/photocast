# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Compute confusion matrices for the realism evaluation

library(jsonlite)
library(tidyverse)

labelers <- c("abbes", "christian", "deborah", "eliane", "yannick")

labels_all_list <- list()
for (ll in seq_along(labelers)) {
  labeler <- labelers[ll]
  parsed <- read_json(paste0("experiment/2021-09-27-realism_evaluation/labels_",
                             labeler, ".json"))

  labels_list <- list()
  for (ii in seq_along(parsed)) {
    labels_list[[ii]] <-
      tibble(judgment = parsed[[ii]]$Label$classifications[[1]]$answer$value,
             file = parsed[[ii]]$`External ID`)
  }
  labels <- bind_rows(labels_list)

  labels <- labels %>%
    mutate(filename = file) %>%
    separate(file, c(NA, "camera", "actual"), "-") %>%
    separate(actual, c("actual", NA)) %>%
    mutate(judgment = ifelse(judgment == "looks_realistic", "real", "generated"),
           camera = substr(camera, 1, 6),
           labeler = labeler)

  labels_all_list[[ll]] <- labels
}

labels_all <- bind_rows(labels_all_list) %>%
  select(camera, filename, actual, judgment, labeler)

labels_all %>% filter(camera == "vrkj06") %>% select(actual, judgment) %>% table()
