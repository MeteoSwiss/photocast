# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

library(lubridate)
library(tidyverse)
library(writexl)

experiment_folder <- "christian/sequence_analog"

files <- list.files(paste0("experiment/2022-06-27-matching_conditions_evaluation/", experiment_folder, "/image"))
images <- substr(files, 1, 20) %>% unique()

load("data/cosmo/surface/cosmo_surface.RData")

cosmo_list <- list()
for (ii in seq_along(images)) {
  image <- images[ii]
  id <- substr(image, 1, 6) %>% toupper()
  ref <- substr(image, 8, 20)

  cosmo_list[[ii]] <- filter(wide, cbs == id &
           reference == as.POSIXct(ref, format = "%Y%m%d_%H%M") %>% floor_date("hours")
  )
}
cosmo <- bind_rows(cosmo_list)

write_xlsx(cosmo, paste0("experiment/2022-06-27-matching_conditions_evaluation/", experiment_folder, "/cosmo.xlsx"))
