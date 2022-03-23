# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Import the COSMO-1 csv files into the wide data.frame

library(tidyverse)

files <- list.files("data/raw__/cosmo_1-2019-2020", "cosmo1_ana_20.*\\.csv",
                    full.names = TRUE)

raw_list <- list()
for (ff in seq_along(files)) {
  print(files[ff])
  raw_list[[ff]] <- read_delim(files[ff], ";", skip = 22, na = "-999.00")
}
raw <- bind_rows(raw_list)

wide <- select(raw, parameter = PARAMETER, reference = DATE_TIME, AXA:YVN) %>%
  pivot_longer(AXA:YVN, names_to = "station", values_to = "value") %>%
  pivot_wider(names_from = "parameter", values_from = "value")

load("data/cam_identification.RData")
wide <- inner_join(wide, identification, c(station = "nat_abbr")) %>%
  select(cbs, station, reference, everything())

save(wide, file = "data/cosmo/surface/cosmo_surface.RData")
