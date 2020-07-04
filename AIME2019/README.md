Author: Chirath Hettiarachchi 2019

This project contains the Type 2 Diabetes Classification carried out using features extracted from PPG Signals.

A description of the project: Hettiarachchi C., Chitraranjan C. (2019) A Machine Learning Approach to Predict Diabetes Using Short Recorded Photoplethysmography and Physiological Characteristics. In: Riaño D., Wilk S., ten Teije A. (eds) Artificial Intelligence in Medicine. AIME 2019. Lecture Notes in Computer Science, vol 11526. Springer, Cham

Main Dataset: Liang,Yongbo,etal."Anew,short-recordedphotoplethysmogramdatasetforbloodpres- sure monitoring in China." Scientific data 5 (2018): 180020.

The extracted PPG features are included in the input folder.
The Matlab scripts to extract the PPG features are in the github repo.
Four experiments are carried out.
* experiment1.py: Healthy vs Diabetes only.
* experiment2.py: Healthy vs Diabetes only and Diabetes with Prehypertension.
* experiment3.py: Healthy vs Diabetes only and Diabetes with Prehypertension / Hypertension.
* experiment4.py: Healthy vs Diabetes only and Diabetes with Prehypertension / Hypertension. (Only PPG Features, Age matched)

The classifiers used are in classifiers.py, where parameter tuning and model evaluation is also carried out.

Late addition:
Specificity values.
