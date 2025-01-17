# Calibrating RAP data with LiDAR-derived ground truth

## Overview
The purpose of this work was to develop a mapped time series of woody cover at MPG Ranch, a conservation property in western Montana, USA. The products will facilitate downstream ecological analyses, such as identificaiton of factors driving bird abundance. One candidate for estimating woody cover is the Rangeland Analysis Platform (RAP; Allred et al. 2021), a machine learning model that predicts plant functional group cover, including shrubs and trees, across the contitental US at a 30m resolution from 1986 to the present. While the spatial and temporal continuity of this product is robust, in practice I found that RAP predictions do not align with LiDAR ground truth at MPG Ranch. This repo contains the code and analyses for a process to calibrate RAP woody cover predictions with LiDAR ground truth at MPG Ranch. I prioritized the use of publically available datasets and libraries for this task so that it may be replicated in other locations.

## Summary of key findings
The uncalibrated RAP map model dramatically overestimates woody cover in upland grassland sites at MPG Ranch. These grassland estimates vary little from adjacent ecologically distinct zones. This presents challenges for ecological analyses seeking to contrast the relative influence of woody cover between, E.G., grasslands and shrubby draws or forested floadplains. The calibration process reduced woody cover overestimation in the grasslands, which represent the vast majority of the study area. The outcome of the calibration process was a strong overall benefit to model performance enabling it to extrapolate well to areas and time periods outside of its trainig data. I observed a 1.26-fold increase in rank order agreement, a 2.8-fold increase in R² score, and a 2.22-fold decrease in mean absolute error relative to uncalibrated RAP predictions in these test areas. Yet, spatially clustered pockets of woody cover error remained in the floodplain, mid elevations regions with dense shrubs, and higher elevations of mixed conifer systems. It is important to note that, while absolute errors remain in some areas, rank order agreement of predictions and ground truth across the landscape were high. This suggests that detecting correlations among woody cover and other ecological meausures may yet prove tractable as areas of relative high and low cover map as exptected to high and low cover areas in LiDAR-derived ground truth.

## Approach
I gathered LiDAR-derived canopy height models (CHMs) produced by the USGS 3D Elevation Project (3DEP; Sugarbaker et al. 2014) and used these as targets for calibrating the RAP predictions. Predictors of LiDAR-derived woody cover included 2019 RAP data, bare earth elevation data (National Elevation Dataset, NED), and terrain derivatives. For model architectures I used LightGBM (Ke et al. 2017), an efficient algorithm for developing predictive models, and stacked it with a general additive model to account for the non-linearity of the relationship between uncalibrated RAP cover and LiDAR ground truth. For model tuning I applied a cross validation approach where hold-out folds were geographically isolated. To evaluate the calibrated models potential to extrapolate to other years and locations, I sourced additional LiDAR data gathered in 2020 located on the southern boundary of MPG Ranch, which served as a test set having no bearing on model design choices. I conducted a suite of analyses to evaluate the performance of the uncalibrated and calibrated models reported below.

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

## Spatial validation strategy
Two LiDAR missions were the focus of this work: one gathered throughout the Bitterroot Valley in 2019, and a second gathered near the northern border of Ravalli County in 2020. These two datasets overlap at the lower reaches of Woodchuck Creek, offering the opportunity to test the performance of this calibration process in a new area and time period. The non-intersecting regions of the 2019 dataset I used as the training set and the basis for model tuning, while the intersecting regions of the 2020 LiDAR data served as a test set to explore the models ability to extrapolate spatially and temporally.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_train_area_designations.png?raw=true" 
       alt="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       title="These areas denote the bounds of areas used for training (green) and testing (magenta) RAP calibration models." 
       width="50%" />
  <br>
  <b>Figure 4:</b> This map depicts the regions used for training (green) and testing (magenta) RAP calibration models.
</p>

I adopted a spatial cross-validaiton strategy for tuning the RAP calibration model, where the basis for hold-out sets were latitudinal bands within the training region ([code](https://github.com/mosscoder/calibrate_rap/blob/main/06_assign_latitudinal_folds.ipynb)). The benefit of assessing latitudinal folds is that they allowed me to assess the models ability to generalize to new areas. For a background in cross validation refer to [these materials](https://machinelearningmastery.com/k-fold-cross-validation/?utm_source=chatgpt.com). Generally, a cross-validation approach is useful for evaluating what model settings, or hyperparameters, allow for optimal learning of a pattern of interest. Note also that I have constrained the training areas to a 500 m buffer surrounding the bird point count sampling locations. This was done to ensure that the product would be most accurate in the areas intended for downstream ecological analyses. Only after model tuning did I predict woody cover for the test set region and evaluate it on the 2020 LiDAR-derived woody cover data to guard against overfitting to the test set.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/folds.png?raw=true" 
       alt="Latitudinal folds" 
       title="Latitudinal folds" 
       width="75%" />
  <br>
  <b>Figure 5:</b> This map depicts the latitudinal bands used as folds during model tuning.
</p>

## Model framework and optimization
For modeling framework I chose a boosted regression tree as implemented in the LightGBM python module (Ke et al. 2017). Boosting is the process of adding models in series, where each model learns from the errors of the previous model. When paired with tree-based models, boosting can effectively model non-linear relationships between predictors and targets. Because ecological cover data are rich in zeros, with a long right tail, careful selection of the objective function (the means by which model error is calculated during learning) is important. I chose the Tweedie objective, as it matches the distribution of the zero-rich cover data (Jørgensen 1987). In practice I found that models fitted with a Tweedie objective correctly represent the rank order of woody cover values, but they tended to underestimate the magnitude of those values, and for this reason I fitted a secondary model, a Generalized Additive Model (GAM; Hastie and Tibshirani 1990), stacked on top of the boosted regression tree predictions to correct for this bias. Stacking is the process of fitting two or more models in series, where the output of the first model is used as the input for the second model.

LightGBM has an abundance of tuning parameters to sort through (Appendix A), and I used a Bayesian optimization approach to sweep across the hyperparameter space to find the optimal settings for the RAP calibration model. I used the optuna python package to perform this search ([code](https://github.com/mosscoder/calibrate_rap/blob/main/07_hyperparam_sweep.ipynb)). Briefly, Bayesian optimization balances exploration and exploitation in a series of trials where new combinations of hyperparameters are tested. Results from prior trials are used as the basis for selecting settings for the next trial. I conducted 150 trials, maximizing the mean of the Normalized Gini Coefficient (assess rank order agreement between predicted and true woody cover) and R2 scores (assess variance explained by the model) across the five folds. Both metrics vary from 0 to 1, where values approaching 1 indicate higher performance. I observed highest performance after 143 trials.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/optimization/optimization_history.png?raw=true" 
       alt="Optimization history" 
       title="Optimization history"/>
  <br>
  <b>Figure 6:</b> Model performance (mean of Normalized Gini Coefficient and R2 scores) improvement after 150 trials of Bayesian optimization.
</p>

## Cross-validated results from the 2019 training set area
We observed performance improvements across all metrics when applying the calibration process to RAP predictions ([code](https://github.com/mosscoder/calibrate_rap/blob/main/08_training_set_eval.ipynb)). The calibrated predictions showed a 1.4-fold increase in rank order agreement (Normalized Gini coefficient), a 3-fold increase in R² score, and a 2.19-fold decrease in mean absolute error. The majority of the study area had low or no woody cover, according to LiDAR-derived ground truth and expert opinion, and in these areas the uncalibrated RAP predictions overestimated woody cover, between 10% and 20%. The calibration process reduced this bias. The uncalibrated RAP model also tended to underestimate woody cover in areas with high LiDAR-derived woody cover, and here again the calibration process reduced bias.

| Metric                    | Uncalibrated RAP | Calibrated RAP |
|--------------------------|------------------|----------------|
| Normalized Gini Coefficient| 0.600           | **0.841**      |
| R² Score                  | 0.210           | **0.629**      |
| Mean Absolute Error       | 14.443          | **6.597**      |


<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/training_true_v_pred.png?raw=true" 
       alt="Training true vs pred" 
       title="Training true vs pred"/>
  <br>
  <b>Figure 7:</b> Scatter plots of LiDAR-derived woody cover vs. uncalibrated RAP predictions (left) and calibrated RAP predictions (right) in 2019.
</p>

By mapping the ground truth, uncalibrated RAP predictions, and calibrated RAP predictions, we can see that the the uncalibrated model dramatically overestimates woody cover in the areas known to be grasslands. 

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/training_set_predictions.png?raw=true" 
       alt="Training predictions" 
       title="Training predictions"/>
  <br>
  <b>Figure 8:</b> Mapped LiDAR-derived woody cover (left), uncalibrated RAP predictions (center), and calibrated RAP predictions (right) in 2019.
</p>

When mapping error across the landscape, we can more clearly see spatial clustering of woody cover errors of both models. While the calibrated model model was able to achieve low error in the grassland areas, areas of denser vegetation were challenging for both models, particularly in the floodplain, mid elevations with dense bitterbrush, and higher elevations of mixed conifer systems. While the absolute error of the calibrated model could be higher in these localized sites, it is important to recognize that rank order agreement (0.84) acoss plant community types was higher than that of the uncalibrated model, and, therefore, detecting correlations among woody cover and other ecological meausures may yet prove tractable.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/training_error_maps.png?raw=true" 
       alt="Training error maps" 
       title="Training error maps"/>
  <br>
  <b>Figure 9:</b> Maps of the error between LiDAR-derived woody cover and uncalibrated RAP predictions (left) and calibrated RAP predictions (right) in 2019. Postive values indicate overestimation, while negative values indicate underestimation.
</p>

## Test set results from 2020 spatiotemporal hold-out areas

I also observed performance improvements, though of lesser magnitude, across all metrics when applying the calibration process to RAP predictions in the 2020 test set ([code](https://github.com/mosscoder/calibrate_rap/blob/main/09_test_set_eval.ipynb)). The calibrated predictions showed a 1.26-fold increase in rank order agreement, a 2.8-fold increase in R² score, and a 2.22-fold decrease in mean absolute error.


| Metric                    | Uncalibrated RAP | Calibrated RAP |
|--------------------------|------------------|----------------|
| Rank Order Agreement| 0.599           | **0.753**      |
| R² Score                  | 0.202           | **0.567**      |
| Mean Absolute Error       | 14.591          | **6.574**      |


<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_true_v_pred.png?raw=true" 
       alt="Test true vs pred" 
       title="Test true vs pred"/>
  <br>
  <b>Figure 10:</b> Scatter plots of LiDAR-derived woody cover vs. uncalibrated RAP predictions (left) and calibrated RAP predictions (right) in 2020.
</p>

As in the training set areas, the uncalibrated model overestimated woody cover in the upland areas. However, LiDAR ground truth showed, in fact, a greater amount of woody cover clustered in certain upland test areas, as compared to similar elevations in the training set areas. Here the calibrated model tended to underestimate woody cover in what I suspect are shrub-dominated areas.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_set_predictions.png?raw=true" 
       alt="Test predictions" 
       title="Test predictions"/>
  <br>
  <b>Figure 11:</b> Mapped LiDAR-derived woody cover (left), uncalibrated RAP predictions (center), and calibrated RAP predictions (right) in 2020.
</p>

Mapping errors across the landscape we can see the regions of error for both models. While the uncalibrated model exhibited error across the region, those of the calibrated model were more localized.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/test_error_maps.png?raw=true" 
       alt="Test error maps" 
       title="Test error maps"/>
  <br>
  <b>Figure 12:</b> Maps of the error between LiDAR-derived woody cover and uncalibrated RAP predictions (left) and calibrated RAP predictions (right) in 2020. Postive values indicate overestimation, while negative values indicate underestimation.
</p>

## Visualizing woody change at bird sampling points
With the time series of predicted cover maps, we can visualize the change in woody cover. I observed most change at the elevation extremes at the bird point count locations, with little change in the mid elevations.

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/figures/cover_change_plots.png?raw=true" 
       alt="Cover change plots" 
       title="Cover change plots"/>
  <br>
  <b>Figure 13:</b> These maps depict predicted woody cover change according to the calibrated RAP model between 2016 and 2011 (left) and 2020 and 2011 (right) at bird point count locations.
</p>

## References
Allred BW, Bestelmeyer BT, Boyd CS, et al. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. Methods Ecol Evol. 2021; 12: 841–849. https://doi.org/10.1111/2041-210X.13564

Hastie, T., & Tibshirani, R. (1990). Generalized Additive Models. Chapman and Hall/CRC.

Jørgensen, B. (1987). Exponential dispersion models. Journal of the Royal Statistical Society. Series B (Methodological), 49(2), 127–162.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), 3149–3157. https://doi.org/10.5555/3294996.3295074

Sugarbaker, L. J., Constance, E. W., Heidemann, H. K., Jason, A. L., Lukas, V., Saghy, D. L., & Stoker, J. M. (2014). The 3D Elevation Program initiative: A call for action (Circular 1399). U.S. Geological Survey. https://doi.org/10.3133/cir1399

## Appendix A: Model tuning hyperparameters

| Hyperparameter           | Description                                                                                      |
|-------------------------|--------------------------------------------------------------------------------------------------|
| `learning_rate`         | The step size at each boosting iteration, balancing convergence speed and accuracy.              |
| `num_leaves`            | The maximum number of leaves in one tree, influencing tree complexity and potential overfitting.  |
| `n_estimators`          | The number of boosting iterations or trees to build. Higher values increase model complexity.     |
| `max_depth`             | The maximum depth of each tree, limiting the number of splits and controlling model complexity.   |
| `tweedie_variance_power`| Parameter controlling the Tweedie distribution shape; values between 1 (Poisson) and 2 (Gamma).  |
| `subsample`             | The fraction of training data used for each boosting iteration, controlling overfitting.          |
| `colsample_bytree`      | The fraction of features used when building each tree, controlling feature sampling.              |
| `reg_alpha`             | L1 regularization term, adding a penalty for non-zero coefficients to encourage sparsity.         |
| `reg_lambda`            | L2 regularization term, adding a penalty for large coefficients to prevent overfitting.           |
| `min_child_samples`     | The minimum number of samples required in a child node to allow further splits.                   |
| `min_child_weight`      | Minimum sum of instance weights (hessian) in a child, preventing splits with insufficient data.   |
| `min_split_gain`        | The minimum gain required for a split to be considered, avoiding small, insignificant splits.     |
| `subsample_freq`        | The frequency (in terms of boosting iterations) to perform subsampling.                           |
| `tpi_ngb`               | Spatial neighborhood in meters used to calculate TPI.                                             |
| `n_splines`             | The number of splines set when fitting a Generalized Additive Model.                             |

<p align="center">
  <img src="https://github.com/mosscoder/calibrate_rap/blob/main/results/optimization/slice_plot.png?raw=true" 
       alt="Slice" 
       title="Slice"/>
  <br>
  <b>Figure A1:</b> Results of Bayesian optimization over the hyperparameter space for the RAP calibration model.
</p>
