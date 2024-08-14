import gradio as gr
import pickle
import pandas as pd
import lftk
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_linguistic_features(text):
    
    doc = nlp(text)
    
    features = ['t_syll3', 'root_propn_var', 'root_space_var', 'corr_punct_var', 'uber_ttr_no_lem', 'a_propn_ps', 'smog']
    LFTK = lftk.Extractor(docs = doc)
    LFTK.customize(stop_words = True, round_decimal = 2)
    doc_features = LFTK.extract(features=features)
    
    X = pd.DataFrame([doc_features])
    
    return X

def predict_text(text):
    
    with open("pipeline.pkl", 'rb') as f:
        pipeline = pickle.load(f)
    
    X_transformed = extract_linguistic_features(text)
    prediction_array = pipeline.predict(X_transformed)
    prediction = prediction_array[0]
    
    result_dict_translation = {0:"Fake 🤬", 1:"True 🤩"}
    
    return result_dict_translation[prediction]


theme = 'freddyaboulton/dracula_revamped'

with gr.Blocks(theme=theme) as text_predictor:
    gr.Markdown("""# Make Believe Fake News Classifier
                English language news classifier based on psycholinguistic and textual coherence features.""")
    with gr.Row():
        news = gr.Textbox(label="News", lines=3, placeholder="Enter a piece of news.")
        outputs = gr.Textbox(label="Output")
    with gr.Row():
        predict_news_button = gr.Button("Process", variant="primary", scale=0.3)
        predict_news_button.click(fn=predict_text, inputs=news, outputs=outputs)
    with gr.Row():
        gr.Markdown(f"For more information about the model development process you can check out the <a href='https://github.com/alberto-lorente/Make_Believe_v2.git'> git repo</a> 🤗.")
        # gr.Markdown("""For more information about the model development process you can check out {git-repo} 🤗.""")

text_predictor.launch()

