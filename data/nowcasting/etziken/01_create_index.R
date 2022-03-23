# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create the master index of images and weather descriptors

library(lubridate)
library(tidyverse)

load("data/cosmo/surface/cosmo_surface.RData")
cosmo <- rename(wide, identifier = cbs) %>%
  mutate(identifier = tolower(identifier))

path <- list.files("data/raw__/image/vrkj37/", ".*\\.jpeg", full.names = TRUE,
                     recursive = TRUE)
base <- basename(path)
images <- tibble(
  identifier = substr(base, 1, 6),
  position = substr(base, 27, 28),
  reference = as.POSIXct(substr(base, 8, 20), format = "%Y%m%d_%H%M"),
  doy = yday(reference),
  hour = hour(reference),
  minute = minute(reference),
  mod = hour*60 + minute,
  reference_floor = reference %>% floor_date("hour"),
  path
)

index <- inner_join(cosmo, images, c("identifier", reference = "reference_floor")) %>%
  select(-reference) %>%
  select(identifier, position, station, reference = reference.y, path, doy, hour, minute, mod, everything())
index = mutate(index, rowid = 0:(nrow(index) - 1), .before = identifier)
save(index, file = "data/nowcasting/etziken/index.RData")
