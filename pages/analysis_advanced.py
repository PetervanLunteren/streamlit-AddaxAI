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
from components import StepperBar, print_widget_label, warning_box, info_box
from utils.common import (
    load_vars, update_vars, clear_vars,
    init_session_state, get_session_var, set_session_var, update_session_vars,
    check_model_availability)
from utils.analysis_utils import (browse_directory_widget,
                                  check_folder_metadata,
                                  project_selector_widget,
                                  datetime_selector_widget,
                                  location_selector_widget, 
                                  cls_model_selector_widget,
                                  det_model_selector_widget,
                                  species_selector_widget,
                                  country_selector_widget,
                                 state_selector_widget,
                                  load_taxon_mapping_cached,
                                  add_deployment_to_queue,
                                  remove_deployment_from_queue,
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
        f"#### Installing virtual environment", key="installing-env", show_close_button=False)
    with modal_install_env.container():
        install_env(get_session_var(
            "analyse_advanced", "required_env_name"))

# modal for downloading models - only create when needed
if get_session_var("analyse_advanced", "show_modal_download_model", False):
    modal_download_model = Modal(
        f"#### Downloading model...", key="download_model", show_close_button=False)
    with modal_download_model.container():
        download_model(get_session_var(
            "analyse_advanced", "download_modelID"), model_meta)

# modal for processing queue - only create when needed
if get_session_var("analyse_advanced", "show_modal_process_queue", False):
    modal_process_queue = Modal(
        f"#### Processing deployments...", key="process_queue", show_close_button=False)
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
        title="#### Describe new project", key="add_project", show_close_button=False)
    with modal_add_project.container():
        add_project_modal()

# modal for adding new location - only create when needed
if get_session_var("analyse_advanced", "show_modal_add_location", False):
    modal_add_location = Modal(
        f"#### Describe new location", key="add_location", show_close_button=False)
    with modal_add_location.container():
        add_location_modal()

# modal for showing classification model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_cls_model_info", False):
    modal_show_cls_model_info = Modal(
        f"#### Model information", key="show_cls_model_info", show_close_button=False)
    with modal_show_cls_model_info.container():
        # Get model info from session state when modal is open
        model_info = get_session_var(
            "analyse_advanced", "modal_cls_model_info_data", {})
        show_cls_model_info_modal(model_info)

# modal for showing none model info - only create when needed
if get_session_var("analyse_advanced", "show_modal_none_model_info", False):
    modal_show_none_model_info = Modal(
        f"#### Model information", key="show_none_model_info", show_close_button=False)
    with modal_show_none_model_info.container():
        show_none_model_info_modal()
 
# modal for folder selector - only create when needed
if get_session_var("analyse_advanced", "show_folder_selector_modal", False):
    modal_folder_selector = Modal(
        title="#### Folder selection", key="folder_selector", show_close_button=False)
    with modal_folder_selector.container():
        folder_selector_modal()

# modal for species selector - only create when needed
if get_session_var("analyse_advanced", "show_modal_species_selector", False):
    modal_species_selector = Modal(
        f"#### Select species", key="species_selector", show_close_button=False)
    with modal_species_selector.container():
        # Get cached data from session state
        nodes = get_session_var("analyse_advanced", "modal_species_nodes", [])
        all_leaf_values = get_session_var( 
            "analyse_advanced", "modal_species_leaf_values", [])
        species_selector_modal(nodes, all_leaf_values)

result = st.segmented_control("Pick one:", ["**No**",
                                   "**Yes**"], key="segmented_control", width="stretch")
st.write("You selected:", result)

st.markdown("*This is where the AI detection happens. Peter will figure this out as this is mainly a  task of rearrangin the previous code.*")

# header
st.header(":material/rocket_launch: Add deployment to database", divider="grey")
st.write(
    "You can analyze one deployment at a time using AI models. "
    "A deployment refers to all the images and videos stored on a single SD card retrieved from the field. "
    "This typically corresponds to one physical camera at one location during a specific period. "
    "The analysis results are saved to a recognition file, which can then be used by other tools in the platform."
)

st.write("")
st.subheader(":material/sd_card: Deployment information", divider="grey")
st.write("Fill in the information related to this deployment. A deployment refers to all the images and videos stored on a single SD card retrieved from the field.")

###### STEPPER BAR ######

st.write("Current step:", step)

# --- Create stepper
stepper = StepperBar(
    steps=["Folder", "Deployment", "Model", "Species"],
    orientation="horizontal",
    active_color="#086164", 
    completed_color="#0861647D",
    inactive_color="#dadfeb"
)
stepper.set_step(step)

# this is the stepper bar that will be used to navigate through the steps of the deployment creation process
with st.container(border=True):

    # stepper bar progress
    st.write("")
    st.markdown(stepper.display(), unsafe_allow_html=True)
    st.divider()

    # folder selection
    if step == 0:

        st.write("Here you can select the folder where your deployment is located. ")

        # select folder
        with st.container(border=True):
            print_widget_label("Folder",
                               help_text="Select the folder where your deployment is located.")
            selected_folder = browse_directory_widget()

            if selected_folder and os.path.isdir(selected_folder):
                check_folder_metadata()
                # st.write(st.session_state)

            if selected_folder and not os.path.isdir(selected_folder):
                st.error(
                    "The selected folder does not exist. Please select a valid folder.")
                selected_folder = None

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        with col_btn_next:
            if selected_folder and os.path.isdir(selected_folder):
                if st.button(":material/arrow_forward: Next", use_container_width=True):
                    # Store selected folder temporarily and advance step
                    set_session_var("analyse_advanced",
                                    "selected_folder", selected_folder)
                    set_session_var("analyse_advanced", "step", 1)
                    st.rerun()
            else:
                st.button(":material/arrow_forward: Next",
                          use_container_width=True,
                          disabled=True,
                          key="project_next_button_dummy")

    elif step == 1:

        with st.container(border=True):
            print_widget_label(
                "Project", help_text="help text")

            selected_projectID = project_selector_widget()

        if selected_projectID:

            # location metadata
            with st.container(border=True):
                print_widget_label(
                    "Location", help_text="help text")
                selected_locationID = location_selector_widget()
            # st.write("")

            # camera ID metadata
            if selected_locationID:
                with st.container(border=True):
                    print_widget_label(
                        "Start", help_text="help text")
                    selected_min_datetime = datetime_selector_widget()

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                # Clear only temporary session state, preserve persistent queue
                clear_vars("analyse_advanced")
                st.rerun()

        if selected_projectID and selected_locationID and selected_min_datetime:
            with col_btn_next:
                if selected_min_datetime:
                    if st.button(":material/arrow_forward: Next", use_container_width=True):
                        # Store selections temporarily and advance step
                        update_session_vars("analyse_advanced", {
                            "step": 2,
                            "selected_projectID": selected_projectID,
                            "selected_locationID": selected_locationID,
                            "selected_min_datetime": selected_min_datetime
                        })
                        # Update persistent general settings with committed projectID
                        update_vars(section="general_settings",
                                    updates={"selected_projectID": selected_projectID})
                        st.rerun()
                else:
                    st.button(":material/arrow_forward: Next",
                              use_container_width=True, disabled=True)

    elif step == 2:
        st.write("MODEL STUFF!")

        needs_installing = False
        selected_cls_modelID = None
        selected_det_modelID = None

        # # load model metadata
        # model_meta = load_model_metadata()

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
                    if st.button(f"Install virtual environment", key="install_cls_virtual_env", use_container_width=False):
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
                    if st.button(f"Download model files", use_container_width=False, key="download_cls_model_button"):
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
                        if st.button(f"Install virtual environment", key="install_det_virtual_env", use_container_width=False):
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
                        if st.button(f"Download model files", use_container_width=False, key="download_det_model_button"):
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
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars(section="analyse_advanced")
                st.rerun()

        with col_btn_next:
            if (selected_cls_modelID and selected_det_modelID) or \
                    (selected_cls_modelID == "NONE" and selected_det_modelID):
                if not needs_installing:
                    if st.button(":material/arrow_forward: Next", use_container_width=True):
                        # Check version compatibility before advancing
                        is_compatible, incompatible_models = check_selected_models_version_compatibility(
                            selected_cls_modelID, selected_det_modelID, model_meta)
                        
                        if is_compatible:
                            # Store model selections temporarily and advance step
                            update_session_vars("analyse_advanced", {
                                "step": 3,
                                "selected_cls_modelID": selected_cls_modelID,
                                "selected_det_modelID": selected_det_modelID
                            })
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
                              use_container_width=True, disabled=True,
                              key="model_next_button_dummy", help="You need to install the required virtual environment or download the model files for the selected models before proceeding. ")
            else:
                st.button(":material/arrow_forward: Next",
                          use_container_width=True, disabled=True,
                          key="model_next_button_dummy", help="You need to select both a species identification model and an animal detection model before proceeding.")

    elif step == 3:

        st.write("Species Selection!")

        selected_cls_modelID = get_session_var(
            "analyse_advanced", "selected_cls_modelID")
        # ✅ OPTIMIZATION 3: Cached taxon mapping loading
        # Only loads CSV file when classification model changes
        # Previous: CSV parsing on every step 3 visit
        # Now: Cached in session state by model ID
        # Handle different model types with their specific requirements
        from components.speciesnet_ui import is_speciesnet_model, render_speciesnet_species_selector
        
        if not selected_cls_modelID == "NONE" and not is_speciesnet_model(selected_cls_modelID):
            # Standard classification models - require species selection
            taxon_mapping = load_taxon_mapping_cached(selected_cls_modelID)
            # st.write(taxon_mapping)
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
                msg="This is where you normally would selectw hich species are present in your project area, but you have not selected a species identification model. Please proceed to add the deployment to the queue.")

        # place the buttons
        col_btn_prev, col_btn_next = st.columns([1, 1])

        # the previous button is always enabled
        with col_btn_prev:
            if st.button(":material/replay: Start over", use_container_width=True):
                clear_vars("analyse_advanced")
                st.rerun()

        with col_btn_next:
            if st.button(":material/playlist_add: Add to queue", use_container_width=True, type="primary"):
                
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

                    # reset step to beginning
                    set_session_var("analyse_advanced", "step", 0)

                    # add deployment to persistent queue
                    add_deployment_to_queue()

                    st.rerun()


# TODO: this shouldnt be in the same var as the other step vars etc., but step etc should be in the session state
# for now its fine, but rename it also that collabroators know that this is not the same as the step vars


st.write("")
st.subheader(":material/traffic_jam: Process queue", divider="grey")
process_queue = analyse_advanced_vars.get("process_queue", [])
if len(process_queue) == 0:
    st.write("You currently have no deployments in the queue. Please add a deployment to the queue to start processing.")
    st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary", disabled=True,
              help="You need to add a deployment to the queue first.")
else:

    st.write(
        f"You currently have {len(process_queue)} deployments in the queue.")
    # col1, _ = st.columns([1, 1])
    # with col1:

    with st.expander(":material/visibility: View queue details", expanded=False):
        with st.container(border=True, height=320):
            for i, deployment in enumerate(process_queue):
                with st.container(border=True):
                    selected_folder = deployment['selected_folder']
                    selected_projectID = deployment['selected_projectID']
                    selected_locationID = deployment['selected_locationID']
                    selected_min_datetime = deployment['selected_min_datetime']
                    selected_det_modelID = deployment['selected_det_modelID']
                    selected_cls_modelID = deployment['selected_cls_modelID']
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
                                     use_container_width=True):
                            remove_deployment_from_queue(i)
                            st.rerun()
                    with col3:
                        # st.popover(f"Process {deployment['selected_folder']}")
                        with st.popover(":material/visibility:", help="Show details", use_container_width=True):
                            # Format all deployment details with consistent styling
                            folder_short = "..." + deployment['selected_folder'][-45:] if len(deployment['selected_folder']) > 45 else deployment['selected_folder']
                            
                            # Get friendly names for models
                            cls_model_name = get_model_friendly_name(deployment['selected_cls_modelID'], 'cls', model_meta)
                            det_model_name = get_model_friendly_name(deployment['selected_det_modelID'], 'det', model_meta)
                            
                            # Display all information in formatted style
                            fields = [
                                ("Folder", folder_short),
                                ("Project", deployment['selected_projectID']),
                                ("Location", deployment['selected_locationID']),
                                ("Species identification model", cls_model_name),
                                ("Animal detection model", det_model_name),
                                ("Selected species", f"{len(deployment['selected_species'])} species" if deployment['selected_species'] else None),
                                ("Country", deployment['selected_country']),
                                ("State", deployment['selected_state']),
                                ("Min datetime", str(deployment['selected_min_datetime']) if deployment['selected_min_datetime'] else None)
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

    if st.button(":material/rocket_launch: Process queue", use_container_width=True, type="primary"):
        # Set session state flag to show modal on next rerun
        set_session_var("analyse_advanced", "show_modal_process_queue", True)
        st.rerun()
