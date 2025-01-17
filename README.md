# Calibrating RAP data with LiDAR-derived ground truth

## Overview
The purpose of this work was to develop a suitable model for woody cover change at MPG Ranch, a conservation property in western Montana, USA. My products will facilitate downstream ecological analyses, such as identificaiton of factors driving bird abundance. One candidate for estimating woody cover change is the Rangeland Analysis Platform (RAP; Allred et al. 2021), a machine learning model that predicts plant functional group cover, including shrubs and trees, across the contitental US at a 30m resolution from 1986 to the present. While the spatial and temporal continuity of this product is robust, in practice I found that RAP predictions do not align with LiDAR ground truth at MPG Ranch. This repo contains the code used to develop a calibration model for correcting RAP woody cover predictions at MPG Ranch.

## Approach
I gathered canopy height models (CHM) derived from USGS 3D Elevation Project LiDAR data (USGS) and used it to calibrate the RAP model. Predictors of LiDAR-derived woody cover included 2019 RAP predictions, bare earth elevation data (National Elevation Dataset, NED), and terrain derivatives. For model architectures I used LightGBM, an efficient algorithm for developing predictive models, and stacked it with a general additive model to account for the non-linearity of the relationship between woody cover and LiDAR metrics. For model tuning I applied a cross validation approach where hold-out folds were geographically isolated. To evaluate the calibrated models potential to extrapolate to other years and locations, I sourced additional LiDAR data gathered in 2020 located on the southern boundary of MPG Ranch, which served as a test set having no bearing on model design choices. I conducted a suite of analyses to evaluate the performance of the uncalibrated and calibrated models reported below.

## Description of data sources and generation of derivative products
To generate woody cover estiamtes from LiDAR data ([code](https://github.com/mosscoder/calibrate_rap/blob/main/01_compute_lidar_cover.ipynb)), I aggregated 1m resolution canopy height data to 30m resolution, counting all pixels above a 1m height threshold to establish canopy cover. This threshold represents an important design choice and trade-off, where alternative values less than 1m risk inclusion of non-woody vegetation, such as tall grassess and forbs, and thresholds greater than 1m risk exclusion of low-lying woody vegetation, such as sagebrush.

I downloaded RAP data from Google Earth Engine ([code](https://github.com/mosscoder/calibrate_rap/blob/main/03_download_rap_from_gee.ipynb)), and summed the shrub and tree woody cover predictions to generate a single woody cover product. Earlier analyses revealed no performance gain from treating shrub and tree cover as separate predictors. I downloaded all available years (1986-2023) for use in generating calibrated predictions, so that I could later assess cover change over time.


<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/rap_woody_cover.png?raw=true" 
       alt="RAP Woody Cover Predictions 2023" 
       title="RAP Woody Cover Predictions 2023" 
       width="50%" />
  <br>
  <b>Figure 1:</b> 2023 woody cover predictions from the uncalibrated RAP model.
</p>

---

I also sourced bare earth elevation data from the USGS 3D Elevation Program ([code](https://github.com/mosscoder/calibrate_rap/blob/main/04_download_ned_from_gee.ipynb)), though I chose the National Elevation Dataset product (10m resolution), rather than LiDAR-derived bare earth elevation data. My concern was that LiDAR-derived bare earth elevation data might contain artifacts betraying canopy properties and therefore presented a risk of data leakage. I resampled these elevation data to match the 30m resolution of the RAP data, and then I generated terrain derivatives ([code](https://github.com/mosscoder/calibrate_rap/blob/main/05_generate_terrain_predictors.ipynb)), including: slope, heat load index (HLI; a proxy of solar inputs), topographic position index (TPI; a measure of a cells elevation relative to its neighbors), and topographic wetness index (TWI; a measure of moisture levels accounting for water flow from uphill areas). I calculated various neighborhoods of topographic position and treated it as a model tuning parameter. I retained the neighborhood corresponding to best performance for the final model. Outside of the TPI neighborhood, I found predictors to be mostly orthogonal, exhibiting no correlations greater than 0.6 in magnitude. This suggests that they each had potential to contribute unique information to the model.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/terrain_fig.png?raw=true"
       alt="Terrain Derivatives" 
       title="Terrain Derivatives" />
  <br>
  <b>Figure 2:</b> Terrain features derived from elevation data used as predictors in calibrating the RAP model. 
</p>

---

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/cormat.png?raw=true" 
       alt="Correlation matrix of terrain predictors" 
       title="Correlation matrix of terrain predictors" />
  <br>
  <b>Figure 3:</b> Correlation matrix of terrain predictors. Values follow tpi_* indicate the spatial neighborhood in meters considered when calculating TPI.
</p>

## Model validation strategy
Two LiDAR missions were the focus of this work: one gathered throughout the Bitterroot Valley in 2019, and a second gathered near the northern border of Ravalli County in 2020. These two datasets overlap at the lower reaches of Woodchuck Creek. The non-intersecting regions of the 2019 dataset I used as the training set and the basis for model tuning, while the intersecting regions of the 2020 LiDAR data served as a test set to explore the models ability to extrapolate to new areas and time periods.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_train_area_designations.png?raw=true" 
       alt="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       title="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       width="50%" />
  <br>
  <b>Figure 4:</b> This map depicts the regions used for training (green) and testing (magenta) RAP calibration models.
</p>

I adopted a five-fold cross-validaiton for selecting the best performing calibration model, where the basis for hold-out folds were latitudinal bands within the training region ([code](https://github.com/mosscoder/calibrate_rap/blob/main/06_assign_latitudinal_folds.ipynb)). 

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/folds.png?raw=true" 
       alt="Latitudinal folds" 
       title="Latitudinal folds" 
       width="50%" />
  <br>
  <b>Figure 5:</b> This map depicts the latitudinal bands used as folds during model tuning.
</p>

## Model fitting

## Model evaluation

## Visualizing woody change at bird sampling points

## References
Allred BW, Bestelmeyer BT, Boyd CS, et al. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. Methods Ecol Evol. 2021; 12: 841–849. https://doi.org/10.1111/2041-210X.13564

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), 3149–3157. https://doi.org/10.5555/3294996.3295074

U.S. Geological Survey. (n.d.). 3D elevation program (3DEP). U.S. Department of the Interior. Retrieved January 17, 2025, from https://www.usgs.gov/core-science-systems/ngp/3dep
