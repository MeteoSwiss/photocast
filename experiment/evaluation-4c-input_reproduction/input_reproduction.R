library(abind)
library(jpeg)
library(lubridate)
library(png)
library(tidyverse)

set.seed(1)
npairs <- 1000

load("data/nowcasting/fluela/index.RData")
tz(index$reference) <- "UTC"

training <- filter(index, reference < as.POSIXct("2020-01-01"))
training_features <- training %>% select(5, 7:37)
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

testing_ref <- index %>% select(reference) %>%
  filter(reference >= as.POSIXct("2020-01-01")) %>%
  filter(hour(reference) >= 6 & hour(reference) <= 14) %>%
  transmute(reference = floor_date(reference, unit = "hours")) %>%
  distinct()

nn <- 1
pixel_distance <- rep(0, npairs)
while(nn <= npairs) {
  print(nn)
  reference_ <- slice_sample(testing_ref, n = 1)$reference

  testing_sequence <- filter(index, reference %in% (reference_ + dhours(0:6))) %>%
    select(7:37) %>% t() %>% matrix(nrow = 1)

  if (length(testing_sequence) != 217)
    next

  testing_sequence <- (testing_sequence - center_est) / scale_est
  delta <- matrix(rep(testing_sequence, each = n), nrow = n) - training_seqfeat
  distance <- apply(delta^2, 1, sum)
  idx <- sort(distance, index.return = TRUE)$ix

  lead_hours <- 0

  testing_path <- filter(index, reference %in% (reference_ + dhours(lead_hours)))$path
  real <- readJPEG(testing_path)

  analog_path <- filter(index, reference %in% (training_sequences[idx[1], ]$reference + dhours(lead_hours)))$path
  analog <- readJPEG(analog_path)

  pixel_distance[nn] <- mean((real - analog)^2)
  nn <- nn + 1
}

mean(pixel_distance)
