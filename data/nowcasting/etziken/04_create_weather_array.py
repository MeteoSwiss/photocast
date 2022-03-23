# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Load and rescale weather descriptors

import numpy as np
import pickle
from pyreadr import read_r
from sklearn import preprocessing

index = read_r('data/nowcasting/etziken/index.RData')['index']

features = index.iloc[:, 6:-1].to_numpy()
scaler = preprocessing.StandardScaler().fit(features)
weather = scaler.transform(features).astype(np.float32)

np.save('data/nowcasting/etziken/weather.npy', weather)
with open('data/nowcasting/etziken/scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle)
