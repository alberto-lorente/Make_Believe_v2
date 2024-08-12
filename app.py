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
    
    # text_utf8 = text.encode("utf-8")
    
    with open("pipeline.pkl", 'rb') as f:
        pipeline = pickle.load(f)
    
    X_transformed = extract_linguistic_features(text)
    prediction_array = pipeline.predict(X_transformed)
    prediction = prediction_array[0]
    
    result_dict_translation = {0:"Fake", 1:"True"}
    
    return result_dict_translation[prediction]


text_predictor = gr.Interface(
    fn=predict_text,
    inputs=["text"],
    outputs=["text"],
)

text_predictor.launch()

