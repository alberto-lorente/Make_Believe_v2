# Make Believe 


**Make Believe** is a project to develop a machine learning solution to detect long-form fake news through the use of __psycholinguistic__, coherence and readibility scores, calculated with [lftk](https://github.com/brucewlee/lftk/blob/main/readme.md#essential-tips-and-to-do-guides). The dataset used to train and evaluate the model can be found [here](https://www.kaggle.com/datasets/hassanamin/textdb3).

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

To illustrate the difference this step made, we can observe the initial correlation of the features in the first image and how many features that were correlated were eliminated after. Notice the decrease in darker colors (highly correlated) in the second image:

![alt text][logo1]

[logo1]: (https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/correlation%20before.png) "Logo Title Text 2"

![alt text][logo2]

[logo2]: (https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/correlation%20after.png) "Logo Title Text 2"

Once the features were trimmed down from 220 to around 40, the features were calculated for the whole dataset. To select the final features, recursive feature analysis was performed and 7 features were selected. 


![alt text][logo3]

[logo3]: (https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/rfe.png) "Logo Title Text 2"

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

Confussion Matrix
![alt text][logo4]

[logo4]: (https://github.com/alberto-lorente/Make_Believe_v2/blob/main/Images%2C%20plots%2C%20graphs/confusion%20matrix.png) "Logo Title Text 2"

## Gradio App

The model is available to play around with in a Gradio app hosted in Hugging Face spaces. Check the [Make Believe App](https://huggingface.co/spaces/alberto-lorente/Make_Believe)!
