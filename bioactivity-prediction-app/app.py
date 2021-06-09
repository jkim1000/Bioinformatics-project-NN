import streamlit as st
import pandas as pd
from PIL import Image
# from rdkit import Chem
# from rdkit.Chem.Draw import IpythonConsole
# from rdkit.Chem import Draw
import numpy as np
# from rdkit.Chem import MACCSkeys, Draw
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def desc_calc(smile):
    # construct the molecule from smile
    mole = Chem.MolFromSmiles(smile)
    # forming MACC fingerprint from the molecue
    macc = MACCSkeys.GenMACCSKeys(mole)
    # convert MACC fp to array
    bitlist = np.asarray(macc)
    return(bitlist)

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Apply model to make predictions
    model = load_model('NN_bioactivity_model.hdf5')
    predicted_pic50 = model.predict(input_data).ravel()
#     prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(predicted_pic50, name='pIC50')
    molecule = pd.Series(load_data['canonical_smiles'], name='molecule')
    df = pd.concat([molecule, prediction_output], axis=1)
    st.write(df)
    
    st.header('**Highest pIC50 compounds**')
    predicted_pic50_df_top3 = df.nlargest(3, 'pIC50')
    st.write(predicted_pic50_df_top3)
    df.to_csv('output.csv')
    
    st.header('**Highest pIC50 compound structure**')
    highest_pic50 = df['molecule'][[17, 8, 9]]
    mols_EGFR = [Chem.MolFromSmiles(smile) for smile in highest_pic50]
    pic50_EGFR = predicted_pic50_df_top3['pIC50'].astype(str).tolist()

    mol = Draw.MolsToGridImage(mols_EGFR, molsPerRow=3, subImgSize=(450, 300), legends=pic50_EGFR)
    st.image(mol)
    
def draw_model(user_input):
    user_input = float(user_input)
    df = pd.read_csv('output.csv')
    molecule = df['molecule'][[user_input]]
    mols_EGFR = [Chem.MolFromSmiles(smile) for smile in molecule]
    pic50_EGFR = df['pIC50'].astype(str).tolist()
    mol = Draw.MolsToGridImage(mols_EGFR, molsPerRow=3, subImgSize=(500, 500), legends=pic50_EGFR)
    return mol

# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Drug Bioactivity Prediction App (Epidermal Growth Factor Receptor)

This app allows you to predict the bioactivity towards targeting the `Epidermal Growth Factor Receptor (EGFR)` protein. 
EGFR is a transmembrane protein activated by binding of its specific ligands and is involved with the cellular growth process. 
It is highly expressed in many cancer cells, thereby making it a target of interest for many cancer drug applications. 


""")

# Sidebar
with st.sidebar.header('1. Upload CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['csv'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_csv(uploaded_file)
    smil_test = load_data['canonical_smiles']
    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):    
        maccs_test_df = pd.DataFrame([desc_calc(smile) for smile in smil_test])

    st.header('**Calculated molecular descriptors**')
    st.write(maccs_test_df)
    st.write(maccs_test_df.shape)


#     # Apply trained model to make prediction on query compounds
    build_model(maccs_test_df)
    


else:
    st.info('Upload input data in the sidebar to start!')
    
with st.sidebar.header('header'):
    user_input = st.text_input('Input index number from the prediction table')
    if user_input:
        with st.spinner("Visualizing..."):
            mol = draw_model(user_input)
        st.image(mol)
        



    
#         user_input = int(user_input)
#         st.header('Visualizing drug structure')
#         highest_pic50 = df['molecule'][user_input]
#         mols_EGFR = [Chem.MolFromSmiles(smile) for smile in highest_pic50]
#         pic50_EGFR = df['pIC50'].astype(str).tolist()

# mols = Draw.MolsToGridImage(mols_EGFR, molsPerRow=3, subImgSize=(450, 300), legends=pic50_EGFR)
# st.image(mols, use_column_width=True)
        
#     if st.sidebar.button('Predict'):
#         load_data = pd.read_csv(uploaded_file)
#         smil_test = load_data['canonical_smiles']
#         st.header('**Original input data**')
#         st.write(load_data)
    
#     else:
#         st.info('Upload input data in the sidebar to start!')
    
#     user_input = st.text_input('Input index number from the prediction table')
#     if user_input:
#         st.header('hello world')
    
# st.header('lul')
