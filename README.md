# SealNet 2.0

---

> U-Net based seal detection model for WV03 panchromatic imagery


---
## Includes:
1. Code to generate training and test sets
2. Utilities for data processing, evaluation, loss functions and training
3. Training loop for a wide hyperparameter search 
4. Unique weighted sampler that helps individual points within training set reach equal probabilities of being sampled when training patches are sampled centered on individual points and may include several neighboring points along with it.