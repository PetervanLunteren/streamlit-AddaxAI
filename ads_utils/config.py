# this file is used to define global variables that are used across the project

import os

AddaxAI_files = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
AddaxAI_streamlit_files = os.path.join(AddaxAI_files, "AddaxAI", "streamlit-AddaxAI") # this is only temporary, will be removed later