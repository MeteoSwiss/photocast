# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create a nowcasting visualization using nearest-neighbor retrieval from the training data

library(abind)
library(jpeg)
library(lubridate)
library(tidyverse)

load("data/nowcasting/fluela/index.RData")
tz(index$reference) <- "UTC"

reference_ <- as.POSIXct("2020-07-02 10:00 UTC")
weight <- c(1, 1, rep(1, 29)) %>% as.matrix() %>% t()  # weighting of individual descriptors

training <- filter(index, reference < as.POSIXct("2020-01-01"))
training_features <- training %>% select(7:37) %>% as.matrix() %>% scale()
n <- nrow(training_features)
center_est <- attr(training_features, "scaled:center")
scale_est <- attr(training_features, "scaled:scale")

composite <- NULL

for (hour in 0:6) {

  ref <- reference_ + dhours(hour)
  target_features <- filter(index, reference == ref) %>%
    select(7:37) %>% as.matrix()
  target_path <- filter(index, reference == ref)$path

  target_features <- (target_features - center_est) / scale_est
  delta <- matrix(rep(target_features, each = n), nrow = n) - training_features

  weight_matrix <- matrix(rep(weight, each = n), nrow = n)
  delta_weighted <- delta^2 * weight_matrix
  distance <- apply(delta_weighted, 1, sum)

  idx <- sort(distance, index.return = TRUE)$ix
  training_paths <- training[idx[1:3], ]$path

  img_row <- NULL
  for (path in c(target_path, training_paths)) {
    img_row <- abind(img_row, readJPEG(path), along = 1)
  }
  composite <- abind(composite, img_row, along = 2)
}

writeJPEG(composite, "experiment/evaluation-nn_retrieval/composite.jpeg")
