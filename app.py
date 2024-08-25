import gradio as gr
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lftk
import spacy
import shap

shap.initjs()

def extract_linguistic_features(text):
    
    doc = nlp(text)
    
    features = ['t_syll3', 'root_propn_var', 'root_space_var', 'corr_punct_var', 'uber_ttr_no_lem', 'a_propn_ps', 'smog']
    LFTK = lftk.Extractor(docs = doc)
    LFTK.customize(stop_words = True, round_decimal = 2)
    doc_features = LFTK.extract(features=features)
    
    X = pd.DataFrame([doc_features])
    
    return X

def predict_text(text):
    
    X_transformed = extract_linguistic_features(text)
    
    prediction_array = pipeline.predict(X_transformed)
    prediction_proba = pipeline.predict_proba(X_transformed)
    prediction_binary = prediction_array[0]
    
    result_dict_translation = {0:"Fake 🤬", 1:"True 🤩"}
    prediction = result_dict_translation[prediction_binary]
    
    return prediction

def explain_prediction(text):
    
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
    explainer = shap.TreeExplainer(model, model_output="raw")
    
    shap_values = explainer(pd.DataFrame(X_explain))

    shap_array = shap_values[0, :, prediction_binary].values
    
    expected_value = round(shap_values[0, :, prediction_binary].base_values, 2)
    shap_contribution = np.sum(shap_array)
    final_value = round(shap_contribution + shap_values[0, :, prediction_binary].base_values, 2)
    
    scores_desc = list(zip(shap_array, cols))
    scores_desc = sorted(scores_desc)
    
    fig = plt.figure()
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title(f"Expected {expected_value} - Model Output {final_value}\n Feature Shap Values")
    plt.ylabel("Feature")
    plt.xlabel("Shap Value")
    
    return fig

theme = 'freddyaboulton/dracula_revamped'

nlp = spacy.load("en_core_web_sm")

with open("pipeline.pkl", 'rb') as f:
    pipeline = pickle.load(f)
    
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
    
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# print(type(pipeline))
# print(type(scaler))
# print(type(model))

with gr.Blocks(theme=theme) as app:
    gr.Markdown("""# Make Believe Fake News Classifier
                English language news classifier based on psycholinguistic and textual coherence features.""")
    with gr.Row():
        news = gr.Textbox(label="News", lines=3, placeholder="Enter a piece of news.")
        outputs = gr.Textbox(label="Output")
        plot = gr.Plot()
    with gr.Row():
        predict_news_button = gr.Button("Process", variant="primary", scale=0.3)
        predict_news_button.click(fn=predict_text, inputs=news, outputs=outputs)
        
        explain_button = gr.Button("Explain", variant="secondary", scale=0.3)
        explain_button.click(fn=explain_prediction, inputs=news, outputs=plot)
        
    with gr.Row():
        gr.Markdown(f"For more information about the model development process you can check out the <a href='https://github.com/alberto-lorente/Make_Believe_v2.git'> git repo</a> 🤗.")
        # gr.Markdown("""For more information about the model development process you can check out {git-repo} 🤗.""")

app.launch()