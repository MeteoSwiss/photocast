# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create all pairs (I_0, w_0) and (I_t, w_t) for training G and D

library(lubridate)
library(tidyverse)
library(RcppCNPy)

lead_time <- 360  # minutes

load("data/nowcasting/fluela/index.RData")

training <- filter(index, reference < as.POSIXct("2020-01-01") &
                     hour >= 5 & hour <= 20)

refindex <- training %>% select(rowid, reference)

pairs_list <- list()
mm_idx <- 1
for (mm in seq(0, lead_time, 10)) {
  pairs_list[[mm_idx]] <- inner_join(
    transmute(refindex, rowid, future = reference + minutes(mm)),
    refindex,
    c(future = "reference")
  ) %>% select(x = rowid.x, y = rowid.y)
  mm_idx <- mm_idx + 1
}
pairs <- bind_rows(pairs_list)

pairs <- pairs[sample(1:nrow(pairs)), ]

npySave("data/nowcasting/fluela/pairs.npy", as.matrix(pairs))
