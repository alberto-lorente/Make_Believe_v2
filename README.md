# Make Believe 


**Make Believe-2016** is a project to develop a machine learning solution to detect long-form fake news related to the 2016 US Presidential Election through the use of __psycholinguistic__, coherence and readibility scores, calculated with [lftk](https://github.com/brucewlee/lftk/blob/main/readme.md#essential-tips-and-to-do-guides). The dataset used to train and evaluate the model can be found [here](https://www.kaggle.com/datasets/hassanamin/textdb3). 

Check the [Make Believe App](https://huggingface.co/spaces/alberto-lorente/Make_Believe)!

## Challenges

The main challenge of the approach is the inherent __colinear__ nature of many of these linguistic features. Let's take a couple of the features found in ltfk for example: 

| Feature        | Description  |
| -------------- |:-------------:|
| n_noun         | total number of nouns|
| n_unoun        | total number of unique nouns |
| t_uword        | total number of unique words |

Intuitively, we can infer that the more unique nouns there are in a text, the more unique words there will be, and viceversa.

In lftk, there are a total of 220 features. This vast number means that compute will be an issue for large amounts of data and there would be redundant, potentially noisy, features in the model if we used all of them.

## Feature selection

Instead of calculating the features for the whole dataset, 400 pieces of news were randomly sampled to calculate the linguistic features effectively and ultimately determine through VIF which features could be omitted altogether before moving on to the feature selection for the model. 

To illustrate the difference this step made, we can observe the initial correlation of the features in the first image and how many features that were correlated were eliminated after.

Notice the amount of squares of darker colors (highly correlated):

![alt text](https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/correlation%20before.png "")

And its decrease in the second image:

![alt text](https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/correlation%20after.png "")

Once the features were trimmed down from 220 to around 40, the features were calculated for the whole dataset. To select the final features, recursive feature analysis was performed and 7 features were selected. 


![alt text](https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/rfe.png "")

These were:

| Feature               | Description  |
| --------------        |:-------------:|
| t_syll3               | total number of words more than three syllables |
| root_propn_var        | root proper nouns variation |
| root_space_var        | root spaces variation |
| corr_punct_var        | corrected punctuations variation |
| uber_ttr_no_lem       | uber type token ratio no lemma |
| a_propn_ps            | average number of proper nouns per sentence |
| smog                  | smog index |

## Model

After playing around with a couple of traditional machine learning models, I settled to use a Random Forest and the hyper-parameters were tuned through a gridsearch. 

The final model had a f-1 weighted score of 0.77.

![alt text](https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/confusion%20matrix.png "")

## Gradio App

The model is available to play around with in a Gradio app hosted in Hugging Face spaces. Check the [Make Believe App](https://huggingface.co/spaces/alberto-lorente/Make_Believe)!
