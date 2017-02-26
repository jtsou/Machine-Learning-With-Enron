# Machine-Learning-With-Enron

The goal of this project is to utilize the financial and email dataset from Enron Corpus, which was made public by US Federal Energy Regulatory Commission during its investigation of Enron, is to establish a model that predicts an individual as a “Person of Interest” (POI). The corpus contains email and financial data of 146 people, most of which are senior management of Enron. The corpus is widely used for various machine learning problems.
The dataset contains 146 records with 18 POI, 128 Non-POI, 21 total features.
The dataset contains missing values and some outliers.
The outliers are:
 TOTAL: Added aggregated data to everything
 THE TRAVEL AGENCY IN THE PARK: this is listed as name
 LOCKHARD EUGENE E: contains only NaN

I used scikit-learn SelectKBest to select the 10 best influential factors. 9 out of 10 features related to financial information are selected. ‘Shared_receipt_with_poi’ is included in this, unsurprisingly. The main purpose of creating this feature, ratio of POI messages, is that we expect POI contact each other more often than non-POIs. And the fact that it is included after using SelectKBest proved that it is quite crucial. The precision score and recall under SVM after the new feature is added went up to [0.4,0.4]. All the selected feature score went up too:
{'to_messages': 1.6463411294420076, 'deferral_payments': 0.22461127473600989, 'exercised_stock_options': 24.815079733218194, 'bonus': 20.792252047181535, 'shared_receipt_with_poi': 8.589420731682381, 'director_fees': 2.1263278020077054, 'from_messages': 0.16970094762175533, 'from_this_person_to_poi': 2.3826121082276739, 'deferred_income': 11.458476579280369, 'from_poi_to_this_person': 5.2434497133749582}

I tried using Random Forest Classifier, Support Vector Machine, GaussianNB, and Logistic
Regression, KMeans and I ended up choosing Support Vector Machine.
Algorithm Precision Recall
Random Forest Classifier 0.33 0.2
Support Vector Machine 0.4 0.4
GaussianNB 0.13 0.8
Logistic Regression 0.33 0.6
KMeans 0.3 0.6

To tune the parameters of the an algorithm means adjusting the algorithm when training it, so
the fit on the test set can be improved. The more tuned the parameter, the more biased the
algorithm will be to the training data. There might be cases of overfitting, which leads to poor
performance.

I tried to tune of algorithm in a way that it is not over fitting, making increment changes to the
parameters. As the result shows, I can get similar results with Logistics Regression and
KMeans. However, Support Vector Machine provides better result without the gap between
recall and precision, which I will explain the significance of both metrics later.
I was hoping GaussianNB would do the trick, but it ended up giving poor performance. The
algorithm also does not allow me to add in parameters such as random_state, which was
frustrating for me.

Validation comprises set of techniques to make sure the models generalize with remaining part
of the dataset. A classic mistake is to overfit the model when it was actually performing well on
training set but poorly on test est. I validated my analysis using cross_validation with 1000 trials.
The trials is inspired by both a project I came across and by tutoring a student college level
statistics. By testing the dataset over and over again, we are able to obtain more correct result.
The test size is 0.3, meaning 3:1 training-to-test ratio.
