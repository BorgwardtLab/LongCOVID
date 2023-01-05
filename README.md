# LongCOVID
A repository to share code for long COVID predictions based on a random forest classifier as well as a univariate association analysis of proteomic features to longCOVID labels.

# Requirements
python 3.7.4 
scikit-learn 1.1.3
pandas 1.5.2
scipy 1.9.3
numpy 1.23.5
shap 0.41.0
statsmodels 0.13.5
openpyxl 3.0.10


# Required input data
The input data required to execute these scripts can be obtained from ![image](https://user-images.githubusercontent.com/54959592/210764179-206b07c3-1845-44e4-9b4f-dcac18ed5de8.png). Please include these in a folder ***Data***. This should comprise: 
- Proteomics_Clinical_Data_220902_Acute_plus_healthy_v5.xlsx
- Proteomics_Clinical_Data_220902_6M_timepoint_v4.xlsx
- Proteomics_Clinical_Data_220902_Labels_v2.xlsx
- Table S2 Biological protein cluster compositions.xlsx


# Execution
We provide the data splits used in ***partitions***. Relevant label dictionaries need to be generated based on the label data file listed above.
Run the file ***prediction_RF.py*** to generate model predictions, 
association analysis either for individual proteomic features, or clusters thereof can be obtained using ***associationAnalysis.py***, and ***associationClusters.py*** respectively. 
In ***combineInterpreations.py*** we combine the SHAP analysis results of multiple cross validation folds. 
