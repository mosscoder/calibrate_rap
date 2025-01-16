# Calibrating RAP data with LiDAR-derived ground truth

## Overview
The purpose of this work was to develop a suitable model for woody cover change at MPG Ranch, a conservation property in western Montana, USA. Our products will facilitate downstream ecological analyses, such as identificaiton of factors driving bird abundance. One candidate for estimating woody cover change is the Rangeland Analysis Platform (RAP; Allred et al. 2021), a machine learning model that predicts plant functional group cover, including shrubs and trees, across the contitental US at a 30m resolution from 1986 to the present. While the spatial and temporal continuity of this product is robust, in practice we found that RAP predictions do not align with LiDAR ground truth at MPG Ranch. This repo contains the code used to develop a calibration model for correcting RAP woody cover predictions at MPG Ranch.

## Approach
We gathered canopy height models (CHM) derived from USGS 3D elevation Project LiDAR data (USGS) and used it to calibrate the RAP model. Predictors of LiDAR-derived woody cover included 2019 RAP predictions, bare earth elevation data (National Elevation Dataset, NED), and terrain derivatives. For model architectures we used LightGBM, an efficient algorithm for developing predictive models, and stacked it with a general additive model to account for the non-linearity of the relationship between woody cover and LiDAR metrics. For model tuning we applied a cross validation approach where hold-out folds were geographically isolated. To evaluate the calibrated models potential to extrapolate to other years and locations, we sourced additional LiDAR data gathered in 2020 located on the southern boundary of MPG Ranch, which served as a test set having no bearing on model design choices. We conducted a suite of analyses to evaluate the performance of the uncalibrated and calibrated models reported below.

## Description of data sources and generation of derivative products
To generate woody cover estiamtes from LiDAR data, we aggregated 1m resolution canopy height data to 30m resolution, counting all pixels above a 1m height threshold to establish canopy cover. This threshold represents an important design choice and trade-off, where alternative values less than 1m risk inclusion of non-woody vegetation, such as tall grassess and forbs, and thresholds greater than 1m risk exclusion of low-lying woody vegetation, such as sagebrush.

We downloaded RAP data from Google Earth Engine, and summed the shrub and tree woody cover predictions to generate a single woody cover product. Earlier analyses revealed no performance gain from treating shrub and tree cover as separate predictors. We downloaded all available years (1986-2023) for use in generating calibrated predictions, so that we could later assess cover change over time.


<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/rap_woody_cover.png?raw=true" 
       alt="RAP Woody Cover Predictions 2023" 
       title="RAP Woody Cover Predictions 2023" 
       width="50%" />
  <br>
  <b>Figure 1:</b> RAP Woody Cover Predictions 2023
</p>

<!-- <figure style="text-align:center;">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/rap_woody_cover.png?raw=true" 
       alt="RAP Woody Cover Predictions 2023" 
       title="RAP Woody Cover Predictions 2023" 
       style="width:50%; height:auto;" />
  <figcaption><b>Figure 1:</b> RAP woody cover predictions 2023</figcaption>
</figure> -->

---

We also sourced bare earth elevation data from the USGS 3D Elevation Program, though we chose the National Elevation Dataset product (10m resolution), rather than LiDAR-derived bare earth elevation data, which might contain artifacts betraying canopy properties and introduce data leakage. We resampled these elevation data to match the 30m resolution of the RAP data, and then we derived terrain derivatives, including slope, heat load index (HLI; a proxy of solar inputs), topographic position index (TPI; a measure of a cells position relative to its neighbors), and topographic wetness index (TWI; a measure of moisture levels accounting for water flow from uphill areas). We calculated various neighborhoods of topographic position and treated it as a model tuning parameter. We retained the neighborhood corresponding to best performance for the final model. Outside of the TPI neighborhood, we found predictors to be mostly orthogonal, exhibiting no correlations greater than 0.6 in magnitude. This suggests that they each had potential to contribute unique information to the model.

<figure style="text-align:center;">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/terrain_fig.png?raw=true" 
       alt="Terrain Derivatives" 
       title="Terrain Derivatives" 
       style="width:50%; height:auto;" />
  <figcaption><b>Figure 2:</b> Terrain Derivatives</figcaption>
</figure>

---

<figure style="text-align:center;">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/cormat.png?raw=true" 
       alt="Correlation matrix of terrain predictors" 
       title="Correlation matrix of terrain predictors" 
       style="width:50%; height:auto;" />
  <figcaption><b>Figure 3:</b> Correlation matrix of terrain predictors</figcaption>
</figure>

## Model validation strategy
We adopted the convention of five-fold cross-validaiton for selecting the best performing calibration model, and tested this on a geographically and temporally distinct test set. It was temporally distinct, in the sense that the LiDAR ground truth were gathered in 2020, during a different scannign mission.

<figure style="text-align:center;">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_train_area_designations.png?raw=true" 
       alt="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       title="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       style="width:50%; height:auto;" />
  <figcaption><b>Figure 4:</b> These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models.</figcaption>
</figure>

## Model fitting

## Model evaluation

## Visualizing woody change at bird sampling points

## References
Allred BW, Bestelmeyer BT, Boyd CS, et al. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. Methods Ecol Evol. 2021; 12: 841–849. https://doi.org/10.1111/2041-210X.13564

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), 3149–3157. https://doi.org/10.5555/3294996.3295074

U.S. Geological Survey. (n.d.). 3D elevation program (3DEP). U.S. Department of the Interior. Retrieved January 17, 2025, from https://www.usgs.gov/core-science-systems/ngp/3dep
