Onlinne Payament Fraud Detection Model

The aim of this project is develop a model that will predict the online payment fraud.
The following steps need to be taken to predict the fraudulent scenario.
Exploratory Data analysis was carried out for proper visualization and understanding data distribution.

>I have derived the dataset from Kaggle.com
>Performed Data analysis by checking Nulll values and duplicates.
>Processed EDA and generated the visualization using various graphs
>Split the data to apply machine learning algorithm

Algorithms used::
1) Logistic Regression
2) Decision Tree Classiifier
3) KNN Classifier

Conclusion:
Fraudsters focus during cashout and transfer mode type transfer.
There is not much information taken from oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest and newbalanceDest columns though they had good positive correlation score.
After applying 3 models, we found that Decision Tree is the best fit model that can be used for this project analysis.
Although the accuracy is 100% in all 3 models, the Precision and recall has improved results in decision tree model.


