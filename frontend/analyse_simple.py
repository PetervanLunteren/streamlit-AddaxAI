import streamlit as st
from backend.analyse_utils import class_selector

# st.set_page_config(layout="wide")

from backend.utils import *
        
# load language settings
txts = load_txts()
vars = load_vars()
lang = vars.get("lang", "en")
mode = vars.get("mode", 1)

# place settings
# settings(txts, lang, mode)
# st.set_page_config(layout="wide")

st.header(":material/rocket_launch: Detect", divider="gray")
st.info("You are currently running AddaxAI in simple mode. This only allows you to deploy models with default settings. To access advanced features, switch to advanced mode in the settings.",
        icon=":material/info:")

# select species
st.subheader(":material/pets: Model selection", divider="gray")
st.write("Which species identification model do you want to use?")
st.selectbox("Select model", ["Model 1", "Model 2", "Model 3"])

# select species
st.subheader(":material/pets: Animal presence", divider="gray")
st.write("Which animals are present in your project area?")
selected = class_selector(["dog", "cat", "bird", "fish", "lizard", "hamster", "turtle", "rabbit", "ferret", "snake", "frog", "tortoise", "gerbil", "chinchilla", "parrot", "canary", "goldfish", "guinea pig", "mouse", "rat"])













# st.selectbox("Select model", ["Model 1", "Model 2", "Model 3"])
# st.button("Deploy model")
# st.write("Model deployed successfully!")
# st.write("Model accuracy: 95%")
# st.write("Model precision: 90%")
# st.write("Model recall: 85%")
# st.write("Model F1 score: 88%")
# st.write("Model ROC-AUC: 92%")
# st.write("Model confusion matrix:")          


# # Define categories and subcategories
# categories = {
#     "Animals": ["Mammals", "Birds", "Reptiles"],
#     "Plants": ["Trees", "Flowers", "Bushes"],
#     "Minerals": ["Rocks", "Ores", "Gemstones"]
# }

# # Select main category
# category = st.selectbox("Select a category", list(categories.keys()))

# # Based on the selected category, show subcategories
# subcategory = st.selectbox(f"Select a subcategory in {category}", categories[category])

# # Display the selected category and subcategory
# st.write(f"You selected: {category} > {subcategory}")