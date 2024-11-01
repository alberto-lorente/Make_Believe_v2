import gradio as gr
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lftk
import spacy
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

def transformer_predict(text):
    
    print("Running Prediction")
    prediction = transformer_pipeline(text)[0]["label"]
    print("Prediction: ", prediction)
    
    return prediction

def explain_transformer_prediction(text):

    prediction = transformer_pipeline(text)[0]["label"]
    prediction_clean = prediction.split(" ")[0]
    binary_prediction = label2id[prediction]
    
    print(f"Prediction {prediction_clean}")
    print(f"Binary Prediction {binary_prediction}")

    shap_values = transformer_explainer([text])
    
    shap_array_values = shap_values[0, :, binary_prediction].values
    shap_array_vocab = shap_values[0, :, binary_prediction].data
    
    print(f"Shap Values {shap_array_values}")
    print(f"Vocab shap {shap_array_vocab}")
    
    shap_dict = dict(zip(shap_array_vocab, shap_array_values))
    sorted_shap_tuples = sorted(shap_dict.items(), key=lambda x:x[1])
    sorted_shap_dict = dict(sorted_shap_tuples)
    
    print(f"Shap Dictionary:\n{sorted_shap_dict}")
    
    sorted_dict_keys = list(sorted_shap_dict.keys())
    sorted_dict_vals = list(sorted_shap_dict.values())
    
    print(f"Sorted vocabulary: {sorted_dict_keys}")
    print(f"Sorted SHAP values: {sorted_dict_vals}")

    fig = plt.figure(tight_layout=True)
    
    plt.barh(sorted_dict_keys, sorted_dict_vals)
    plt.title(f"Feature Contribution to the {prediction_clean} prediction")
    plt.ylabel("Feature")
    plt.xlabel("Shap Value")
    
    return fig
    
def extract_linguistic_features(text):
    
    doc = nlp(text)
    
    features = ['t_syll3', 'root_propn_var', 'root_space_var', 'corr_punct_var', 'uber_ttr_no_lem', 'a_propn_ps', 'smog']
    LFTK = lftk.Extractor(docs = doc)
    LFTK.customize(stop_words = True, round_decimal = 2)
    print("Extracting Features...")
    doc_features = LFTK.extract(features=features)
    
    X = pd.DataFrame([doc_features])
    print("Features extracted!")
    return X

def predict_text_ml(text):
    
    print("Running Prediction")

    X_transformed = extract_linguistic_features(text)
    
    prediction_array = pipeline.predict(X_transformed)
    prediction_proba = pipeline.predict_proba(X_transformed)
    prediction_binary = prediction_array[0]
    
    result_dict_translation = {0:"Fake 🤬", 1:"True 🤩"}
    prediction = result_dict_translation[prediction_binary]
    print("Prediction: ", prediction)

    return prediction



def explain_ml_prediction(text):
    
    X_transformed = extract_linguistic_features(text)
    
    prediction_array = pipeline.predict(X_transformed)
    prediction_binary = prediction_array[0]
    
    scaled_X = scaler.transform(X_transformed)
    
    cols = ['words_>_3_syllables',
            'root_proper_nouns_var',
            'root_spaces_var',
            'corrected_punctuations_var',
            'uber_type_token_ratio_no_lemma',
            'avg_PPN_per_sentence',
            'smog_index']
    
    X_explain = pd.DataFrame(scaled_X, columns=cols)
    
    shap_values = ml_explainer(pd.DataFrame(X_explain))

    shap_array = shap_values[0, :, prediction_binary].values
    
    expected_value = round(shap_values[0, :, prediction_binary].base_values, 2)
    shap_contribution = np.sum(shap_array)
    final_value = round(shap_contribution + shap_values[0, :, prediction_binary].base_values, 2)
    
    scores_desc = list(zip(shap_array, cols))
    scores_desc = sorted(scores_desc)
    
    fig = plt.figure(tight_layout=True)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title(f"Expected {expected_value} - Model Output {final_value}\n Feature Shap Values")
    plt.ylabel("Feature")
    plt.xlabel("Shap Value")
    
    return fig

theme = 'freddyaboulton/dracula_revamped'

# for hf model
model_name = "alberto-lorente/distilbert-make-believe16"

label2id = {"True 🤩": 1, "Fake 🤬":0}
id2label = {value: key for key, value in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                        num_labels=2, 
                                                        id2label=id2label, 
                                                        label2id=label2id)
transformer_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
transformer_explainer = shap.Explainer(transformer_pipeline)

# for ml model
nlp = spacy.load("en_core_web_sm")

with open("pipeline.pkl", 'rb') as f:
    pipeline = pickle.load(f)
    
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
    
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
ml_explainer = shap.TreeExplainer(model, model_output="raw")

def predict(text, model):
    
    if model == "distilbert-make-believe16":
        prediction = transformer_predict(text)

        return prediction

    elif model == "rf-make-believe16":
        prediction = predict_text_ml(text)
        
        return prediction
    
def explain(text, model):
    
    if model == "distilbert-make-believe16":
        explanation_fig = explain_transformer_prediction(text)
        
        return explanation_fig
    
    elif model == "rf-make-believe16":
        explanation_fig = explain_ml_prediction(text)
        
        return explanation_fig
    
with gr.Blocks(theme=theme) as app:
    
    gr.Markdown("""# Project Make Believe-2016: US Election Fake News Detection
                English language news classifiers trained on news related to the 2016 US Presidential Election.""")
    
    model_choice = gr.Radio(["distilbert-make-believe16", "rf-make-believe16"], label="Select a model", info="Which model would you like to use? **distilbert-make-believe16** is a finetuned DistilBert model while **rf-make-believe16** is a random forest model trained via linguistic features computed with lftk. SHAP explanations are available for both!")
    
    with gr.Row():
        
        news = gr.Textbox(label="News", lines=3, placeholder="Enter a piece of news.")
        outputs = gr.Textbox(label="Output")
        plot = gr.Plot(label="SHAP Explanation")
        
    with gr.Row():
        
        predict_news_button = gr.Button("Process", variant="primary", scale=0.3)
        explain_button = gr.Button("Explain", variant="secondary", scale=0.3)
        
        predict_news_button.click(fn=predict, inputs=[news, model_choice], outputs=outputs)
        explain_button.click(fn=explain, inputs=[news, model_choice], outputs=plot)

    with gr.Row():
        gr.Markdown(f"For more information about the model development process you can check out the <a href='https://github.com/alberto-lorente/Make_Believe_v2.git'> git repo</a> 🤗.")
        # gr.Markdown("""For more information about the model development process you can check out {git-repo} 🤗.""")

app.launch()