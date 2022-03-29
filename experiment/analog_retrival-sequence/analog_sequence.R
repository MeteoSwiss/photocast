# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create a nowcasting visualization using analog retrieval of image sequences from the training data

library(abind)
library(jpeg)
library(lubridate)
library(png)
library(tidyverse)

load("data/nowcasting/fluela/index.RData")
tz(index$reference) <- "UTC"

reference_ <- as.POSIXct("2020-07-02 10:00 UTC")
weight <- c(rep(rep(1, 31), 7)) %>% as.matrix() %>% t()
# weight <- c(rep(rep(1, 2), 7)) %>% as.matrix() %>% t()

training <- filter(index, reference < as.POSIXct("2020-01-01"))

training_features <- training %>% select(5, 7:37)
# training_features <- training %>% select(5, 7:8)
training_sequences <- training_features
for (h in 1:6) {
  training_sequences <- inner_join(
    training_sequences,
    mutate(training_features, reference = reference - dhours(h)),
    "reference"
  )
}
training_seqfeat <- training_sequences %>% select(-1) %>% as.matrix() %>% scale()

n <- nrow(training_seqfeat)
center_est <- attr(training_seqfeat, "scaled:center")
scale_est <- attr(training_seqfeat, "scaled:scale")

testing_sequence <- filter(index, reference %in% (reference_ + dhours(0:6))) %>%
  select(7:37) %>% t() %>% matrix(nrow = 1)
testing_sequence <- (testing_sequence - center_est) / scale_est
delta <- matrix(rep(testing_sequence, each = n), nrow = n) - training_seqfeat
weight_matrix <- matrix(rep(weight, each = n), nrow = n)
delta_weighted <- delta^2 * weight_matrix
distance <- apply(delta_weighted, 1, sum)
idx <- sort(distance, index.return = TRUE)$ix

testing_paths <- filter(index, reference %in% (reference_ + dhours(0:6)))$path
img_row <- NULL
for (path in testing_paths) {
  img_row <- abind(img_row, readJPEG(path), along = 2)
}
composite <- img_row

analog_paths <- filter(index, reference %in% (training_sequences[idx[1], ]$reference + dhours(0:6)))$path
img_row <- NULL
for (path in analog_paths) {
  img_row <- abind(img_row, readJPEG(path), along = 2)
}
composite <- abind(composite, img_row, along = 1)

writePNG(composite, "experiment/analog_retrival-sequence/composite.png")


