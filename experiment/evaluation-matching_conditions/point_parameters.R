# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# retrieve COSMO-1 weather descriptor values

library(lubridate)
library(tidyverse)

load("data/cosmo/surface/cosmo_surface.RData")

image <- "vrkj44_20200425_1200"
id <- substr(image, 1, 6) %>% toupper()
ref <- substr(image, 8, 20)

filter(wide, cbs == id &
         reference == as.POSIXct(ref, format = "%Y%m%d_%H%M") %>% floor_date("hours")
       ) %>% View()
