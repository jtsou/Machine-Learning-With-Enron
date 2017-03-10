# Machine-Learning-With-Enron

<h2>The goal </h2> <p>This project is to utilize the financial and email dataset from Enron Corpus, which was made public by US Federal Energy Regulatory Commission during its investigation of Enron, is to establish a model that predicts an individual as a “Person of Interest” (POI). The corpus contains email and financial data of 146 people, most of which are senior management of Enron. The corpus is widely used for various machine learning problems.</p>

<p> I used precision and recall as 2 main evaluation metrics. The algorithm of my choosing ‘GaussianNB’ produced a precision of 0.5 and a recall score of 0.6.</p>

<p>Precision refers to ratio of true positive – predicted POI matches actual result.
Recall refers to ratio of true positive of people flagged as POI. In English, my result indicated that if the model predicts 100 POIs, there would be 50 people that are actually POIs and the rest of 50 are not. With recall score of 0.6, the model finds 60% of all real POIs in prediction. This model is good at finding bad guys without missing anyone. </p>
