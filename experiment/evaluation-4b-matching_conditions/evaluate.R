# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

library(tidyverse)
library(readxl)
library(boot)

set.seed(1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/christian/gan/evaluation.xlsx",
                    na = "NA")
cg <- sheet %>% select(c(1:7, 9, 10)) %>%
  mutate(camera = substr(image, 1, 6), expert = "christian", algorithm = "gan") %>%
  select(-1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/christian/sequence_analog/evaluation.xlsx",
                    na = "NA")
ca <- sheet %>% select(c(1, 3:8, 10, 11)) %>%
  mutate(camera = substr(image, 1, 6), expert = "christian", algorithm = "analog") %>%
  select(-1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/flavia/gan/evaluation_rev.xlsx",
                    na = "NA")
fg <- sheet %>% select(c(1:7, 9, 10)) %>%
  mutate(camera = substr(image, 1, 6), expert = "flavia", algorithm = "gan") %>%
  select(-1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/flavia/sequence_analog/evaluation_rev.xlsx",
                    na = "NA")
fa <- sheet %>% select(c(1, 3:8, 10, 11)) %>%
  mutate(camera = substr(image, 1, 6), expert = "flavia", algorithm = "analog") %>%
  select(-1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/therese/gan/evaluation.xlsx",
                    na = "NA")
tg <- sheet %>% select(c(1, 3:8, 10, 11)) %>%
  mutate(camera = substr(image, 1, 6), expert = "therese", algorithm = "gan") %>%
  select(-1)

sheet <- read_excel("experiment/2022-06-27-matching_conditions_evaluation/therese/sequence_analog/evaluation.xlsx",
                    na = "NA")
ta <- sheet %>% select(c(1, 3:8, 10, 11)) %>%
  mutate(camera = substr(image, 1, 6), expert = "therese", algorithm = "analog") %>%
  select(-1)

evaluation <- rbind(cg, ca, fg, fa, tg, ta)


# Consistency of examiners ------------------------------------------------

evaluation %>% filter(algorithm == "gan") %>%
  group_by(expert) %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()

evaluation %>% filter(algorithm == "analog") %>%
  group_by(expert) %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()


# Method means ------------------------------------------------------------

evaluation %>% filter(algorithm == "gan") %>%
  group_by(camera) %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()

evaluation %>% filter(algorithm == "gan") %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()

evaluation %>% filter(algorithm == "analog") %>%
  group_by(camera) %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()

evaluation %>% filter(algorithm == "analog") %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>% print()


# Method CIs per camera --------------------------------------------------------

statistic <- function(data, indices) {
  subset <- data[indices, ] %>% as.matrix()
  apply(subset, 2, function(x) mean(x, na.rm = TRUE))
}

ci_list <- list()
rr <- 1
for (algorithm in unique(evaluation$algorithm)) {
  for (camera in unique(evaluation$camera)) {
    boot_analysis <- evaluation %>%
      filter(algorithm == !!algorithm & camera == !!camera) %>%
      select(where(is.numeric)) %>%
      boot(statistic, 10000)
    for (index in 1:8) {
      print(rr)
      ci <- boot.ci(boot_analysis, type = "bca", index = index)
      if (is.null(ci))
        next
      ci_list[[rr]] <- tibble(algorithm, camera, condition = attr(ci$t0, "names"),
                              lower = ci$bca[4], upper = ci$bca[5])
      rr <- rr + 1
    }
  }
}
ci <- bind_rows(ci_list)
ci %>% arrange(algorithm, camera, condition) %>% View()


# Method CIs overall --------------------------------------------------------

ci_list <- list()
rr <- 1
for (algorithm in unique(evaluation$algorithm)) {

  boot_analysis <- evaluation %>%
    filter(algorithm == !!algorithm) %>%
    select(where(is.numeric)) %>%
    boot(statistic, 10000)
  for (index in 1:8) {
    print(rr)
    ci <- boot.ci(boot_analysis, type = "bca", index = index)
    if (is.null(ci))
      next
    ci_list[[rr]] <- tibble(algorithm,  condition = attr(ci$t0, "names"),
                            lower = ci$bca[4], upper = ci$bca[5])
    rr <- rr + 1
  }
}
ci <- bind_rows(ci_list)
ci %>% arrange(algorithm, condition) %>% View()


# Cloud cover at Cevio example --------------------------------------------

cc <- evaluation %>% filter(camera == "vrkj44" & algorithm == "gan")
print(mean(cc$`cloud cover`))
print(sum(cc$`cloud cover`))

cc %>% filter(`cloud cover` == 0 & `w_t accurate` == 1 & `Î_t consistent` == 0) %>%  View()
