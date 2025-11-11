"""
AddaxAI Advanced Analysis Tool

4-step wizard for processing camera trap deployments:
1. Folder Selection: Choose deployment folder
2. Deployment Metadata: Project, location, capture dates  
3. Model Selection: Detection + classification AI models
4. Species Selection: Target species for analysis area

Features optimized startup with session state caching and write-through configuration updates.
"""

import streamlit as st
import os
from st_modal import Modal

from utils.config import *


# import local modules
from components import StepperBar, print_widget_label, warning_box, info_box, code_span
from utils.common import (
    load_vars, update_vars, clear_vars,
    init_session_state, get_session_var, set_session_var, update_session_vars,
    check_model_availability)
from utils.analysis_utils import (browse_directory_widget,
                                  check_folder_metadata,
                                #   project_selector_widget,
                                  datetime_selector_widget,
                                  location_selector_widget, 
                                  cls_model_selector_widget,
                                  det_model_selector_widget,
                                  species_selector_widget,
                                  country_selector_widget,
                                 state_selector_widget,
                                  load_taxon_mapping_cached,
                                  add_run_to_queue,
                                  remove_run_from_queue,
                                  get_model_friendly_name,
                                  install_env,
                                  run_process_queue,
                                  download_model,
                                  add_project_modal,
                                  add_location_modal,
                                  show_cls_model_info_modal,
                                  show_none_model_info_modal,
                                  species_selector_modal,
                                  folder_selector_modal,
                                  check_selected_models_version_compatibility
                                  )

# Page config
st.set_page_config(layout="centered")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED RESOURCE LOADING - Use cached session state values from main.py
# ═══════════════════════════════════════════════════════════════════════════════

# ✅ OPTIMIZATION 1: Use cached resources from main.py session state
# Eliminates 3 expensive file reads per rerun:
# - load_lang_txts() -> txts (language JSON file)
# - load_model_metadata() -> model_meta (large model metadata JSON)
# - load_vars("general_settings") -> lang/mode (settings JSON file)
txts = st.session_state["txts"]
model_meta = st.session_state["model_meta"]
lang = st.session_state["shared"]["lang"]
mode = st.session_state["shared"]["mode"]

# Load persistent vars (needs fresh data for queue management)
analyse_advanced_vars = load_vars(section="analyse_advanced")
general_settings_vars = load_vars(section="general_settings")

# Ensure shared session state has the selected project (important for restarts)
if "shared" not in st.session_state:
    st.session_state["shared"] = {}
if "selected_projectID" not in st.session_state["shared"]:
    st.session_state["shared"]["selected_projectID"] = general_settings_vars.get("selected_projectID", None)

# Initialize session state for this tool
init_session_state("analyse_advanced")

# Initialize step from session state
step = get_session_var("analyse_advanced", "step", 0)



# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY MODALS - Now optimized to only create when needed using session state
# ═══════════════════════════════════════════════════════════════════════════════

# modal for installing environment - only create when needed
if get_session_var("analyse_advanced", "show_modal_install_env", False):
    modal_install_env = Modal(
        f"#### Installing virtual environment",
        key="installing-env",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_install_env.container():
        install_env(get_session_var(
            "analyse_advanced", "required_env_name"))

# modal for downloading models - only create when needed
if get_session_var("analyse_advanced", "show_modal_download_model", False):
    modal_download_model = Modal(
        f"#### Downloading model...",
        key="download_model",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_download_model.container():
        download_model(get_session_var(
            "analyse_advanced", "download_modelID"), model_meta)

# modal for processing queue - only create when needed
if get_session_var("analyse_advanced", "show_modal_process_queue", False):
    modal_process_queue = Modal(
        f"#### Processing runs...",
        key="process_queue",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_process_queue.container():
        # Process queue should always be loaded from persistent storage
        process_queue = analyse_advanced_vars.get("process_queue", [])
        run_process_queue(process_queue)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED MODAL MANAGEMENT - Only create modals when needed using session state
# ═══════════════════════════════════════════════════════════════════════════════
# This approach prevents expensive modal creation on every rerun by using session
# state flags to conditionally create modals only when they should be displayed.

# modal for adding new project - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_project", False):
    modal_add_project = Modal(
        title="#### Describe new project",
        key="add_project",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_add_project.container():
        add_project_modal()

# modal for adding new location - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_location", False):
    modal_add_location = Modal(
        f"#### Describe new location",
        key="add_location",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_add_location.container():
        add_location_modal()

# modal for showing classification model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_cls_model_info", False):
    modal_show_cls_model_info = Modal(
        f"#### Model information",
        key="show_cls_model_info",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_show_cls_model_info.container():
        # Get model info from session state when modal is open
        model_info = get_session_var(
            "analyse_advanced", "modal_cls_model_info_data", {})
        show_cls_model_info_modal(model_info)

# modal for showing none model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_none_model_info", False):
    modal_show_none_model_info = Modal(
        f"#### Model information",
        key="show_none_model_info",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_show_none_model_info.container():
        show_none_model_info_modal()
 
# modal for folder selector - only create when needed
if get_session_var("analyse_advanced", "show_folder_selector_modal", False):
    modal_folder_selector = Modal(
        title="#### Folder selection",
        key="folder_selector",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_folder_selector.container():
        folder_selector_modal()

# modal for species selector - only create when needed
if get_session_var("analyse_advanced", "show_modal_species_selector", False):
    modal_species_selector = Modal(
        f"#### Select species",
        key="species_selector",
        show_close_button=False,
        show_title=False,
        show_divider=False
    )
    with modal_species_selector.container():
        # Get all available species and current selection
        all_species = get_session_var("analyse_advanced", "modal_all_species", [])
        selected_species = get_session_var("analyse_advanced", "selected_nodes", [])
        species_selector_modal(all_species, selected_species)

st.subheader(":material/add: Add run to queue", divider="grey")
st.write("Fill in the information related to this run below and add to the queue for batch processing. It is advised to run the analysis per camera deployment, as many of the subsequent tools depend on deployment metadata. You can add multiple runs to the queue and process them in one go.")

###### STEPPER BAR ######

# --- Create dynamic stepper based on deployment type and classification model selection
last_deployment_type = general_settings_vars.get("last_deployment_type", True)
is_deployment = get_session_var("analyse_advanced", "is_deployment", last_deployment_type)

# Check if classification model is selected (affects whether species step is shown)
selected_cls_modelID = get_session_var("analyse_advanced", "selected_cls_modelID", None)

# If no model selected in session state yet, check for project's preferred model
if selected_cls_modelID is None:
    from utils.analysis_utils import get_cached_map, DEFAULT_CLASSIFICATION_MODEL
    selected_projectID = st.session_state["shared"].get("selected_projectID")
    
    if selected_projectID:
        # Load project-specific preferred classification model
        map_data, _ = get_cached_map()
        project_data = map_data["projects"].get(selected_projectID, {})
        preferred_models = project_data.get("preferred_models", {})
        selected_cls_modelID = preferred_models.get("cls_model", DEFAULT_CLASSIFICATION_MODEL)
    else:
        # Fallback to global default
        selected_cls_modelID = general_settings_vars.get("selected_modelID", DEFAULT_CLASSIFICATION_MODEL)

has_classification_model = selected_cls_modelID and selected_cls_modelID != "NONE"

# Build stepper steps dynamically
stepper_steps = ["Data"]

if is_deployment:
    stepper_steps.append("Deployment")

stepper_steps.append("Model")

if has_classification_model:
    stepper_steps.append("Species")

# Map internal step numbers to stepper display
if is_deployment and has_classification_model:
    # Full 4-step wizard: Data, Deployment, Model, Species
    current_stepper_step = step
elif is_deployment and not has_classification_model:
    # 3-step wizard: Data, Deployment, Model (no species)
    current_stepper_step = step
elif not is_deployment and has_classification_model:
    # 3-step wizard: Data, Model, Species (skip deployment)
    if step == 0:
        current_stepper_step = 0  # Data
    elif step == 2:
        current_stepper_step = 1  # Model (internal step 2 -> display step 1)
    elif step == 3:
        current_stepper_step = 2  # Species (internal step 3 -> display step 2)
    else:
        current_stepper_step = 0  # Fallback
else:
    # 2-step wizard: Data, Model (skip deployment and species)
    if step == 0:
        current_stepper_step = 0  # Data
    elif step == 2:
        current_stepper_step = 1  # Model (internal step 2 -> display step 1)
    else:
        current_stepper_step = 0  # Fallback

stepper = StepperBar(
    steps=stepper_steps,
    orientation="horizontal",
    active_color="#086164", 
    completed_color="#0861647D",
    inactive_color="#dadfeb"
)
stepper.set_step(current_stepper_step)

# this is the stepper bar that will be used to navigate through the steps of the run creation process
with st.container(border=True):

    # stepper bar progress
    st.write("")
    st.markdown(stepper.display(), unsafe_allow_html=True)
    st.divider()

    # data selection (folder, project, deployment type)
    if step == 0:

        

        # Get current project directly from session state (which is updated by sidebar)
        selected_projectID = get_session_var("shared", "selected_projectID", None)
        
        # Store it in analyse_advanced for later use in subsequent steps
        if selected_projectID:
            set_session_var("analyse_advanced", "selected_projectID", selected_projectID)
        
        if not selected_projectID:
            # No project selected - show message to create first project
            info_box("No project selected. Create your first project using the sidebar. Each run needs to fall under a project.")
        else:
                    
            # folder selection
            with st.container(border=True):
                print_widget_label("Which data do you want to analyse?",
                                help_text="Select the folder where your data is located.")
                selected_folder = browse_directory_widget()

                if selected_folder and os.path.isdir(selected_folder):
                    check_folder_metadata()
                    # st.write(st.session_state)

                if selected_folder and not os.path.isdir(selected_folder):
                    st.error(
                        "The selected folder does not exist. Please select a valid folder.")
                    selected_folder = None
            
            # deployment type selection
            with st.container(border=True):
                print_widget_label(
                    "Is this a single camera trap deployment?",
                    help_text="A deployment refers to data from a camera trap placed at a specific location. Select 'No' if this is just a folder of images without location context."
                )
                
                # Define callback to update session state
                def on_deployment_type_change():
                    if "is_deployment_control" in st.session_state:
                        set_session_var("analyse_advanced", "is_deployment", st.session_state["is_deployment_control"])
                
                # Get current selection or load last used value from persistent storage
                last_deployment_type = general_settings_vars.get("last_deployment_type", True)
                is_deployment = get_session_var("analyse_advanced", "is_deployment", last_deployment_type)
                
                # Create segmented control with callback
                deployment_options = {True: "Yes", False: "No"}
                segmented_contorl_col1, _ = st.columns([1, 3])
                with segmented_contorl_col1:
                    st.segmented_control(
                        "Is this data of a single camera trap deployment?",
                        options=list(deployment_options.keys()),
                        format_func=deployment_options.get,
                        default=is_deployment,
                        label_visibility="collapsed",
                        key="is_deployment_control",
                        on_change=on_deployment_type_change,
                        width="stretch"
                        )

            # place the buttons
            col_btn_prev, col_btn_next = st.columns([1, 1])

            with col_btn_next:
                if selected_folder and os.path.isdir(selected_folder):
                    if st.button(":material/arrow_forward: Next", width='stretch'):
                        # Get current deployment type setting
                        last_deployment_type = general_settings_vars.get("last_deployment_type", True)
                        is_deployment = get_session_var("analyse_advanced", "is_deployment", last_deployment_type)
                        selected_projectID = get_session_var("analyse_advanced", "selected_projectID")
                        
                        # Store selected project and folder temporarily
                        update_session_vars("analyse_advanced", {
                            "selected_projectID": selected_projectID,
                            "selected_folder": selected_folder,
                        })
                        
                        # Update persistent general settings with committed projectID
                        update_vars(section="general_settings",
                                    updates={"selected_projectID": selected_projectID})
                        
                        # Skip step 1 for non-deployment data, go directly to step 2 (models)
                        if is_deployment:
                            # Go to step 1 (deployment metadata)
                            set_session_var("analyse_advanced", "step", 1)
                        else:
                            # Skip step 1, go directly to step 2 (models)
                            # Set default values for deployment metadata
                            update_session_vars("analyse_advanced", {
                                "selected_locationID": None,
                                "selected_min_datetime": None,
                                "step": 2
                            })
                        
                        st.rerun()
                else:
                    st.button(":material/arrow_forward: Next",
                            width='stretch',
                            disabled=True,
                            key="data_next_button_dummy")

    elif step == 1:
                
        # Get deployment type and project from session state (set in step 0)
        is_deployment = get_session_var("analyse_advanced", "is_deployment", True)
        selected_projectID = get_session_var("analyse_advanced", "selected_projectID")
        
        # Show normal metadata collection for deployments
        
        # location metadata
        with st.container(border=True):
            print_widget_label(
                "Where was the camera located?", help_text="help text")
            selected_locationID = location_selector_widget()
        # st.write("")

        # camera ID metadata
        if selected_locationID:
            with st.container(border=True):
                print_widget_label(
                    "What was the start of this deployment?", help_text="help text")
                selected_min_datetime = datetime_selector_widget()

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", width='stretch'):
                # Clear only temporary session state, preserve persistent queue
                clear_vars("analyse_advanced")
                st.rerun()

        # Adjust next button logic based on deployment type
        if is_deployment:
            # For deployments, require location and datetime (project already selected in step 0)
            can_proceed = selected_locationID and selected_min_datetime
        else:
            # For non-deployments, no additional requirements (project already selected in step 0)
            can_proceed = True
            
        if can_proceed:
            with col_btn_next:
                if st.button(":material/arrow_forward: Next", width='stretch'):
                    # Store deployment metadata and advance step
                    update_session_vars("analyse_advanced", {
                        "step": 2,
                        "selected_locationID": selected_locationID,
                        "selected_min_datetime": selected_min_datetime
                    })
                    st.rerun()
        else:
            with col_btn_next:
                st.button(":material/arrow_forward: Next",
                          width='stretch', disabled=True)

    elif step == 2:

        needs_installing = False
        selected_cls_modelID = None
        selected_det_modelID = None

        # select cls model
        with st.container(border=True):
            print_widget_label("Species identification model",
                               help_text="Here you can select the model of your choosing.")
            selected_cls_modelID = cls_model_selector_widget(model_meta)

            if selected_cls_modelID and selected_cls_modelID != "NONE":

                # Check model availability (simple filesystem checks)
                availability = check_model_availability(
                    'cls', selected_cls_modelID, model_meta)

                # Check environment availability
                if not availability['env_exists']:
                    needs_installing = True
                    warning_box(title="Virtual environment required",
                                msg=f"The selected model needs a specific virtual environment that is not yet installed. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.",
                                icon=":material/warning:")
                    if st.button(f"Install virtual environment", key="install_cls_virtual_env", width='content'):
                        set_session_var(
                            "analyse_advanced", "required_env_name", availability['env_name'])
                        # Set session state flag to show modal on next rerun
                        set_session_var("analyse_advanced",
                                        "show_modal_install_env", True)
                        st.rerun()

                # Check model file availability
                if not availability['model_exists']:
                    needs_installing = True
                    warning_box(
                        title="Model download required",
                        msg=f"The selected classification model still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                    if st.button(f"Download model files", width='content', key="download_cls_model_button"):
                        set_session_var(
                            "analyse_advanced", "download_modelID", selected_cls_modelID)
                        # Set session state flag to show modal on next rerun
                        set_session_var("analyse_advanced",
                                        "show_modal_download_model", True)
                        st.rerun()
        # st.write("")

        # select detection model
        if selected_cls_modelID or selected_cls_modelID != "NONE":
            with st.container(border=True):
                print_widget_label("Animal detection model",
                                   help_text="The species identification model you selected above requires a detection model to locate the animals in the images. Here you can select the model of your choosing.")
                selected_det_modelID = det_model_selector_widget(model_meta)
                if selected_det_modelID:

                    # Check model availability (simple filesystem checks)
                    availability = check_model_availability(
                        'det', selected_det_modelID, model_meta)

                    # Check environment availability
                    if not availability['env_exists']:
                        needs_installing = True
                        warning_box(title="Virtual environment required",
                                    msg=f"The selected model needs a specific virtual environment that is not yet installed. Please install it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.",
                                    icon=":material/warning:")
                        if st.button(f"Install virtual environment", key="install_det_virtual_env", width='content'):
                            set_session_var(
                                "analyse_advanced", "required_env_name", availability['env_name'])
                            # Set session state flag to show modal on next rerun
                            set_session_var("analyse_advanced",
                                            "show_modal_install_env", True)
                            st.rerun()

                    # Check model file availability
                    if not availability['model_exists']:
                        needs_installing = True
                        warning_box(
                            title="Model download required",
                            msg=f"The selected classification model still needs to be downloaded. Please download it before proceeding. This is a one-time setup step and may take a few minutes, depending on your internet speed.")
                        if st.button(f"Download model files", width='content', key="download_det_model_button"):
                            set_session_var(
                                "analyse_advanced", "download_modelID", selected_det_modelID)
                            # Set session state flag to show modal on next rerun
                            set_session_var("analyse_advanced",
                                            "show_modal_download_model", True)
                            st.rerun()

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", width='stretch'):
                clear_vars(section="analyse_advanced")
                st.rerun()

        with col_btn_next:
            if (selected_cls_modelID and selected_det_modelID) or \
                    (selected_cls_modelID == "NONE" and selected_det_modelID):
                if not needs_installing:
                    # Check if we should show "Next" or "Add to queue" button
                    if selected_cls_modelID == "NONE":
                        # No classification model - skip species selection, go directly to queue
                        button_text = ":material/playlist_add: Add to queue"
                        button_type = "primary"
                    else:
                        # Classification model selected - proceed to species selection
                        button_text = ":material/arrow_forward: Next"
                        button_type = "secondary"
                    
                    if st.button(button_text, width='stretch', type=button_type):
                        # Check version compatibility before proceeding
                        is_compatible, incompatible_models = check_selected_models_version_compatibility(
                            selected_cls_modelID, selected_det_modelID, model_meta)
                        
                        if is_compatible:
                            # Store model selections temporarily
                            update_session_vars("analyse_advanced", {
                                "selected_cls_modelID": selected_cls_modelID,
                                "selected_det_modelID": selected_det_modelID
                            })
                            
                            if selected_cls_modelID == "NONE":
                                # No classification model - add to queue directly
                                # Set default values for species selection
                                set_session_var("analyse_advanced", "selected_species", None)
                                set_session_var("analyse_advanced", "selected_country", None) 
                                set_session_var("analyse_advanced", "selected_state", None)
                                
                                # Save deployment type preference to general settings
                                current_is_deployment = get_session_var("analyse_advanced", "is_deployment", True)
                                update_vars("general_settings", {"last_deployment_type": current_is_deployment})
                                
                                # Reset step to beginning
                                set_session_var("analyse_advanced", "step", 0)
                                
                                # Add run to persistent queue
                                add_run_to_queue()
                            else:
                                # Classification model selected - proceed to species selection
                                set_session_var("analyse_advanced", "step", 3)
                            
                            st.rerun()
                        else:
                            # Show warning without rerunning
                            for model in incompatible_models:
                                warning_box(
                                    title=f"Update required for {model['name']}",
                                    msg=f"Minimum version <code style='color:#086164; font-family:monospace;'>v{model['required_version']}</code> required, but you have <code style='color:#086164; font-family:monospace;'>v{model['current_version']}</code>. Please visit https://addaxdatascience.com/addaxai/#install to update."
                                )
                else:
                    st.button(":material/arrow_forward: Next",
                              width='stretch', disabled=True,
                              key="model_next_button_dummy", help="You need to install the required virtual environment or download the model files for the selected models before proceeding. ")
            else:
                st.button(":material/arrow_forward: Next",
                          width='stretch', disabled=True,
                          key="model_next_button_dummy", help="You need to select both a species identification model and an animal detection model before proceeding.")

    elif step == 3:

        selected_cls_modelID = get_session_var(
            "analyse_advanced", "selected_cls_modelID")
        # cached taxon mapping loading

        from components.speciesnet_ui import is_speciesnet_model, render_speciesnet_species_selector
        
        if not selected_cls_modelID == "NONE" and not is_speciesnet_model(selected_cls_modelID):
            # Standard classification models - require species selection
            taxon_mapping = load_taxon_mapping_cached(selected_cls_modelID)
            with st.container(border=True):
                print_widget_label("Species presence",
                                   help_text="Here you can select the model of your choosing.")
                selected_species = species_selector_widget(
                    taxon_mapping, selected_cls_modelID)
                # No country/state needed for standard models
                selected_country = None
                selected_state = None
        elif is_speciesnet_model(selected_cls_modelID):
            
            # SpeciesNet models - require country selection, no species selection
            selected_species, selected_country, selected_state = render_speciesnet_species_selector()
                
        else:
            # No classification model selected
            selected_species = None
            selected_country = None
            selected_state = None
            info_box(
                title="No species identification model selected",
                msg="This is where you normally would select which species are present in your project area, but you have not selected a species identification model. Please proceed to add the run to the queue.")

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", width='stretch'):
                clear_vars("analyse_advanced")
                st.rerun()

        with col_btn_next:
            if st.button(":material/playlist_add: Add to queue", width='stretch', type="primary"):
                
                # Validate model requirements
                from components.speciesnet_ui import validate_speciesnet_requirements
                
                model_id = selected_cls_modelID or ""  # guard against None
                is_none_model = (model_id == "NONE")
                is_speciesnet = is_speciesnet_model(model_id)
                needs_species = bool(model_id) and not is_none_model and not is_speciesnet 

                # Validation logic
                validation_passed = True
                
                if is_speciesnet:
                    validation_passed = validate_speciesnet_requirements(selected_country)
                elif needs_species and not selected_species:
                    warning_box(
                        msg="At least one species must be selected for species classification.",
                        title="Species selection required"
                    )
                    validation_passed = False
                
                if validation_passed:
                    # store selected species (will be empty for SPECIESNET or NONE, which is fine)
                    set_session_var("analyse_advanced", "selected_species", selected_species)

                    # Save deployment type preference to general settings
                    current_is_deployment = get_session_var("analyse_advanced", "is_deployment", True)
                    update_vars("general_settings", {"last_deployment_type": current_is_deployment})

                    # reset step to beginning
                    set_session_var("analyse_advanced", "step", 0)

                    # add run to persistent queue
                    add_run_to_queue()

                    st.rerun()


st.write("")
st.subheader(":material/traffic_jam: Process queue", divider="grey")
process_queue = analyse_advanced_vars.get("process_queue", [])
if len(process_queue) == 0:
    st.write("You currently have no runs in the queue. Please add a run to the queue to start processing.")
    st.button(":material/rocket_launch: Process queue", width='stretch', type="primary", disabled=True,
              help="You need to add a run to the queue first.")
else:

    st.markdown(
        f"You currently have {code_span(len(process_queue))} runs in the queue.", unsafe_allow_html=True)

    with st.expander(":material/visibility: View queue details", expanded=False):
        with st.container(border=True, height=320):
            for i, run in enumerate(process_queue):
                with st.container(border=True):
                    selected_folder = run['selected_folder']
                    selected_projectID = run['selected_projectID']
                    selected_locationID = run['selected_locationID']
                    selected_min_datetime = run['selected_min_datetime']
                    selected_det_modelID = run['selected_det_modelID']
                    selected_cls_modelID = run['selected_cls_modelID']
                    col1, col2, col3 = st.columns([6, 1, 1])

                    with col1:
                        folder_short = "..." + \
                            selected_folder[-45:] if len(
                                selected_folder) > 45 else selected_folder
                        text = f"Folder &nbsp;&nbsp;<code style='color:#086164; font-family:monospace;'>{folder_short}</code>"
                        st.markdown(
                            f"""
                                <div style="background-color: #f0f2f6; padding: 7px; border-radius: 8px;">
                                    &nbsp;&nbsp;{text}
                                </div>
                                """,
                            unsafe_allow_html=True
                        )

                    with col2:
                        if st.button(":material/delete:", help="Remove from queue", key=f"remove_{i}",
                                     width='stretch'):
                            remove_run_from_queue(i)
                            st.rerun()
                    with col3:
                        with st.popover(":material/visibility:", help="Show details", width='stretch'):
                            # Format all run details with consistent styling
                            folder_short = "..." + run['selected_folder'][-45:] if len(run['selected_folder']) > 45 else run['selected_folder']
                            
                            # Get friendly names for models
                            cls_model_name = get_model_friendly_name(run['selected_cls_modelID'], 'cls', model_meta)
                            det_model_name = get_model_friendly_name(run['selected_det_modelID'], 'det', model_meta)
                            
                            # Display all information in formatted style
                            fields = [
                                ("Folder", folder_short),
                                ("Project", run['selected_projectID']),
                                ("Location", run['selected_locationID']),
                                ("Species identification model", cls_model_name),
                                ("Animal detection model", det_model_name),
                                ("Selected species", f"{len(run['selected_species'])} species" if run['selected_species'] else None),
                                ("Country", run['selected_country']),
                                ("State", run['selected_state']),
                                ("Start", run['selected_min_datetime'].replace('T', ' ') if run['selected_min_datetime'] else None),
                                ("Number of images", run.get('n_images')),
                                ("Number of videos", run.get('n_videos'))
                            ]
                            
                            # Only show fields that have values
                            for field_name, field_value in fields:
                                if field_value:
                                    st.markdown(
                                        f"""
                                        <div style="margin-bottom: 3px;">
                                            <strong>{field_name}:</strong> <code style='color:#086164; font-family:monospace;'>{field_value}</code>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

    if st.button(":material/rocket_launch: Process queue", width='stretch', type="primary"):
        # Set session state flag to show modal on next rerun
        set_session_var("analyse_advanced", "show_modal_process_queue", True)
        st.rerun()
