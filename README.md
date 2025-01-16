# Calibrating RAP data with LiDAR

## Overview
The purpose of this work is to develop a suitable model for woody cover change at MPG Ranch, a conservation property in western Montana, USA. Our products will facilitate downstream analyses, such as identificaiton of factors driving bird abundance. One candidate is the Rangeland Analysis Platform (RAP; Allred et al. 2021), a machine learning model that predicts plant functional group cover, including shrub and tree woody cover, across the contitental US at a 30m resolution from 1986 to the present. While the spatial and temporal continuity of this product is robust, in practice we found that RAP predictions do not align with LiDAR ground truth at MPG Ranch. This repo contains the code used to develop a calibration model for RAP woody cover predictions at MPG Ranch.

## Approach
We gathered canopy height models (CHM) derived from USGS 3D elevation Project LiDAR data (USGS) and used it to calibrate the RAP model. Inputs included 2019 RAP predictions, woody cover estimates from 2019 LiDAR ground truth, bare earth elevation data (National Elevation Dataset, NED), and terrain derivatives. For model architectures we used LightGBM, an efficient algorithm for developing predictive models, and stacked it with a general additive model to account for the non-linearity of the relationship between woody cover and LiDAR metrics. For model tuning we applied a cross validation approach. To evaluate the calibrated models potential to extrapolate to other years and locations, we sourced additional LiDAR data gathered in 2020, which served as a test set. We provide a suite of analyses to evaluate the performance of the uncalibrated and calibrated models.

## Description of data sources


## Model validation strategy
![test_train_area_designations](https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_train_area_designations.png?raw=true "These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models.")

## Model fitting

## Model evaluation

## Visualizing woody change at bird sampling points

## References
Allred BW, Bestelmeyer BT, Boyd CS, et al. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. Methods Ecol Evol. 2021; 12: 841–849. https://doi.org/10.1111/2041-210X.13564

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), 3149–3157. https://doi.org/10.5555/3294996.3295074

U.S. Geological Survey. (n.d.). 3D elevation program (3DEP). U.S. Department of the Interior. Retrieved January 17, 2025, from https://www.usgs.gov/core-science-systems/ngp/3dep
