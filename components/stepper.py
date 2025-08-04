"""
Stepper bar component for AddaxAI Streamlit application.
"""


class StepperBar:
    """
    A visual stepper component to show progress through multiple steps.
    
    Supports both horizontal and vertical orientations with customizable colors.
    """
    
    def __init__(self, steps, orientation='horizontal', active_color='blue', completed_color='green', inactive_color='gray'):
        self.steps = steps
        self.step = 0
        self.orientation = orientation
        self.active_color = active_color
        self.completed_color = completed_color
        self.inactive_color = inactive_color

    def set_step(self, step):
        """Set the current active step."""
        if 0 <= step < len(self.steps):
            self.step = step
        else:
            raise ValueError("Step index out of range")

    def display(self):
        """Display the stepper bar."""
        if self.orientation == 'horizontal':
            return self._display_horizontal()
        elif self.orientation == 'vertical':
            return self._display_vertical()
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")

    def _display_horizontal(self):
        """Display horizontal stepper bar."""
        stepper_html = "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        for i, step in enumerate(self.steps):
            if i < self.step:
                icon = "check_circle"
                color = self.completed_color
            elif i == self.step:
                icon = "radio_button_checked"
                color = self.active_color
            else:
                icon = "radio_button_unchecked"
                color = self.inactive_color

            stepper_html += f"""
            <div style='text-align:center;'>
                <span class="material-icons" style="color:{color}; font-size:30px;">{icon}</span>
                <div style="color:{color};">{step}</div>
            </div>"""
            if i < len(self.steps) - 1:
                stepper_html += f"<div style='flex-grow:1; height:2px; background-color:{self.inactive_color};'></div>"
        stepper_html += "</div>"
        return stepper_html

    def _display_vertical(self):
        """Display vertical stepper bar."""
        stepper_html = "<div style='display:flex; flex-direction:column; align-items:flex-start;'>"
        for i, step in enumerate(self.steps):
            color = self.completed_color if i < self.step else self.inactive_color
            current_color = self.active_color if i == self.step else color
            stepper_html += f"""
            <div style='display:flex; align-items:center; margin-bottom:10px;'>
                <div style='width:30px; height:30px; border-radius:50%; background-color:{current_color}; margin-right:10px;'></div>
                <div>{step}</div>
            </div>"""
            if i < len(self.steps) - 1:
                stepper_html += f"<div style='width:2px; height:20px; background-color:{self.inactive_color}; margin-left:14px;'></div>"
        stepper_html += "</div>"
        return stepper_html