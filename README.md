Make Believe 

The main goal of this project is to create a Celebrity Fake News detection model based on the linguistic differences between true and fake news. The dataset I will be using to develop the model is Web Celebrity, a database introduced by Verónica Rosas-Pérez et al. in her 2018 paper "Automatic Detection of Fake News".

I will also be using Text Characterization Toolkit (TCT) by Daniel Simig et al., a coh-metrix style feature computing tool. In short, TCT extracts discursive, syntactic, lexical and descriptive characteristics from the input text making it an interesting feature engineering tool. Aside from the TCT, I will also be using NLTK and Textblob to extract sentiment scores for the texts, as it is common in fake news detection. 

Summary of the project development

After extracting the TCT features and sentiment scores for the texts and scaling them to make them more easily interpretable, I looked for ways to filter out features since the TCT alone computes more than 65 and some of them may be redundat. The sets of features I chose to extract were 1. based in the correlation between feature-Polarity, 2. performing the Kolmogorov-Smirnov test to check whether two samples (sample of feature x for the True class and sample for feature x for the Fake class) come from the same population, 3. a set obtained through recursive feature elimination and, finally, 4. a set with all the available features (Part 3). 

I trained 4 baseline models (SVC, Random Forest, K-Neighbors and Logistic Regression) for each of the 4 sets of features and compared the results to determine which pair of set of features-ml algorithm performed best for the task. The F1 score was the primary metric for this evaluation (Part 4). After this step, it was clear that going with a random forest algorithm trained on the rfe set would make the most sense (Part 5).

For fine-tuning, I used gridsearch to look for the optimal depth, number of estimators and split criterion function. Since there were going to be many models fitted and a lot are not that far off in terms of F1 score I also computed the ROC AUC (Part 6).

Looking at the gridsearch results, the performance of the models started to stabilize at around a depth of 8, the number of estimators did not make much difference and the gini index was the best performing criterion. Since the f1 score of the models with these characteristics did not vary that much, I prioritized choosing a model that would be less complex so that we would also avoid overfitting if possible. The final model had a maximum depth of 8, 150 estimators and the split quality algorithm chosen was the gini index (Part 7).

The final model was a Random Forest trained on the hyperparameters previously mention with the rfe set. The final F1 score was close to 91% with a ROC AUC close to 90%. As a little exercise, I picked a different dataset from Kaggle, performed the same data transformations we carried out on the first dataset and used the final model to predict all the non-NaN samples. This validation set came to be 5524 samples and the F1 score obtained by the model was a tiny bit over 70% (Part 8).


Thank you for checking out the project and I hope you found something interesting or useful in it :)
 
