# Calibrating RAP data with LiDAR

## Overview
The purpose of this work is to develop a suitable model for woody cover change at MPG Ranch, a conservation property in western Montana, USA. Our products will facilitate development of downstream analyses, such as identificaiton of factors driving bird abundance. The Rangeland Analysis Platform (RAP; Allred et al. 2021) is a machine learning model that predicts plant functional group cover, including shrub and tree woody cover, across the contitental US at a 30m resolution from 1986 to the present. While the spatial and temporal continuity of this product is robust, in practice we found that RAP predictions do not align with LiDAR ground truth at MPG Ranch. This repo contains the code used to develop a calibration model for RAP woody cover predictions at MPG Ranch.

## Approach
We gathered canopy height models (CHM) derived from LiDAR data as part of the USGS 3DEP project (3DEP citation) and used it to calibrate the RAP model. Inputs include 2019 RAP predictions, 2019 LiDAR ground truth, elevation, and terrain derivatives. To calibrate RAP woody cover predictions, we used a LightGBM model. 

## Data sources

## Model validation strategy
![test_train_area_designations](https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_train_area_designations.png?raw=true =100x "These areas denote the bounds of areas used for training and testing RAP calibration models.")

## Model fitting

## Model evaluation

## Visualizing woody change at bird sampling points

## References
Allred BW, Bestelmeyer BT, Boyd CS, et al. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. Methods Ecol Evol. 2021; 12: 841–849. https://doi.org/10.1111/2041-210X.13564

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), 3149–3157. https://doi.org/10.5555/3294996.3295074

USGS 3DEP LiDAR
