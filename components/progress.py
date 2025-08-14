"""
Progress bar components for AddaxAI Streamlit application.
"""

import streamlit as st
import re


class MultiProgressBars:
    """
    A container for managing multiple progress bars with different states and labels.
    
    Supports both manual updates and automatic updates from tqdm objects or strings.
    """
    
    def __init__(self, container_label="Progress Bars"):
        self.container = st.container(border=True)
        self.label_placeholder = self.container.empty()
        if container_label:
            self.label_placeholder.markdown(container_label)
        self.bars = {}
        self.states = {}
        self.max_values = {}
        self.active_prefixes = {}
        self.wait_labels = {}
        self.pre_labels = {}
        self.done_labels = {}
        self.show_device = {}
        self.device_info = {}
        self.label_divider = " \u00A0\u00A0 | \u00A0\u00A0 "

    def update_label(self, new_label):
        """Update the container label dynamically."""
        if new_label:
            self.label_placeholder.markdown(new_label)
        else:
            self.label_placeholder.empty()

    def add_pbar(self, label, max_value=None, show_device=False, wait_text="Waiting...", pre_text="Starting...", running_text="Running...", done_text="Done"):
        """Add a new progress bar to the container."""
        container = self.container.container()
        pbar_id = label
        
        self.states[pbar_id] = 0
        self.max_values[pbar_id] = max_value or 1  # temporary placeholder
        self.show_device[pbar_id] = show_device
        self.device_info[pbar_id] = None
        
        # Generate phase labels from base label with custom text
        wait_label = f"**{label}**{self.label_divider}{wait_text}"
        pre_label = f"**{label}**{self.label_divider}{pre_text}"
        active_prefix = f"**{label}**{self.label_divider}{running_text}"
        done_label = f"**{label}**{self.label_divider}{done_text}"
        
        self.active_prefixes[pbar_id] = active_prefix
        self.wait_labels[pbar_id] = wait_label
        self.pre_labels[pbar_id] = pre_label
        self.done_labels[pbar_id] = done_label
        
        # Show wait_label initially
        self.bars[pbar_id] = container.progress(0, text=wait_label)

    def reset_pbar(self, pbar_id):
        """Reset a progress bar to 0 and show wait_label."""
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")
        
        self.states[pbar_id] = 0
        self.bars[pbar_id].progress(0, text=self.wait_labels[pbar_id])

    def reset_all_pbars(self):
        """Reset all progress bars to 0 and show wait_labels."""
        for pbar_id in self.bars:
            self.reset_pbar(pbar_id)

    def start_pbar(self, pbar_id):
        """Transition from wait_label to pre_label state."""
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")
        
        # Reset state and show pre_label
        self.states[pbar_id] = 0
        self.bars[pbar_id].progress(0, text=self.pre_labels[pbar_id])

    def set_max_value(self, pbar_id, max_value):
        """Set the maximum value for a progress bar."""
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")
        self.max_values[pbar_id] = max_value
        self.states[pbar_id] = 0
        # If we have a wait_label and haven't started yet, keep showing it
        if self.wait_labels[pbar_id] and self.states[pbar_id] == 0:
            self.bars[pbar_id].progress(0, text=self.wait_labels[pbar_id])
        else:
            self.bars[pbar_id].progress(0, text=self.pre_labels[pbar_id])

    def update(self, pbar_id, n=1, text=""):
        """Update a progress bar by incrementing its value."""
        if pbar_id not in self.bars:
            raise ValueError(f"Progress bar '{pbar_id}' not found.")

        self.states[pbar_id] += n
        if self.states[pbar_id] > self.max_values[pbar_id]:
            self.states[pbar_id] = self.max_values[pbar_id]

        progress = self.states[pbar_id] / self.max_values[pbar_id]
        display_text = (
            self.done_labels[pbar_id]
            if self.states[pbar_id] >= self.max_values[pbar_id]
            else f"{self.active_prefixes[pbar_id]} {text}".strip()
        )

        self.bars[pbar_id].progress(progress, text=display_text)
        
    def update_from_tqdm_string(self, pbar_id, tqdm_line: str):
        """Parse a tqdm output string and update the corresponding Streamlit progress bar, including ETA."""
        # Check for GPU availability info if show_device is enabled
        if pbar_id in self.show_device and self.show_device[pbar_id] and self.device_info[pbar_id] is None:
            if "GPU available: True" in tqdm_line:
                self.device_info[pbar_id] = "GPU"
            elif "GPU available: False" in tqdm_line:
                self.device_info[pbar_id] = "CPU"
        
        tqdm_pattern = r"(\d+)%\|.*\|\s*(\d+)/(\d+).*?\[(.*?)<([^,]+),\s*([\d.]+)\s*(\S+)?/s\]"
        match = re.search(tqdm_pattern, tqdm_line)

        if not match:
            return  # Skip lines that do not match tqdm format

        percent = int(match.group(1))
        n = int(match.group(2))
        total = int(match.group(3))
        elapsed_str = match.group(4).strip()
        eta_str = match.group(5).strip()
        rate = float(match.group(6))
        unit = match.group(7) or ""

        self.set_max_value(pbar_id, total)
        self.states[pbar_id] = n  # Sync directly to avoid increment error

        # Build label components
        label_parts = [
            f":material/clock_loader_40: {percent}%",
            f":material/laps: {n} {unit} / {total} {unit}",
            f":material/speed: {rate:.2f} {unit}/s",
            f":material/timer: {elapsed_str}",
            f":material/sports_score: {eta_str}"
        ]
        
        # Add device info if available and enabled
        if (pbar_id in self.show_device and self.show_device[pbar_id] and 
            pbar_id in self.device_info and self.device_info[pbar_id]):
            device_label = f":material/memory: {self.device_info[pbar_id]}"
            label_parts.insert(0, device_label)
        
        label = self.label_divider + self.label_divider.join(label_parts)

        self.update(pbar_id, n - self.states[pbar_id], text=label)

    def update_from_tqdm_object(self, pbar_id, pbar):
        """Update the progress bar directly from a tqdm object."""
        if pbar_id not in self.bars:
            return
        
        fmt = pbar.format_dict
        n = fmt.get("n", 0)
        total = fmt.get("total", 1)
        rate = fmt.get("rate")
        unit = fmt.get("unit", "B")
        elapsed = fmt.get("elapsed")
        

        def fmt_time(s):
            if s is None:
                return ""
            s = int(s)
            return f"{s // 60}:{s % 60:02}"
        
        def fmt_bytes(bytes_val, suffix="B"):
            """Format bytes into human readable format"""
            if bytes_val is None or bytes_val == 0:
                return f"0 {suffix}"
            elif bytes_val < 1024:
                return f"{bytes_val:.0f} {suffix}"
            elif bytes_val < 1024**2:
                return f"{bytes_val/1024:.1f} K{suffix}"
            elif bytes_val < 1024**3:
                return f"{bytes_val/(1024**2):.1f} M{suffix}"
            else:
                return f"{bytes_val/(1024**3):.1f} G{suffix}"

        # Update max value if needed
        if self.max_values[pbar_id] != total:
            self.max_values[pbar_id] = total
        
        # Calculate progress (allow over 100% like you wanted)
        progress = min(n / total, 1.0) if total > 0 else 0
        
        # Generate label with icons and proper units
        percent = int(n / total * 100) if total > 0 else 0
        percent_str = f":material/clock_loader_40: {percent}%"
        
        # Format current/total - only format bytes if unit is "B"
        if unit == "B":
            n_formatted = fmt_bytes(n)
            total_formatted = fmt_bytes(total)
            laps_str = f":material/laps: {n_formatted} / {total_formatted}"
            rate_formatted = fmt_bytes(rate, 'B/s') if rate else ""
            rate_str = f":material/speed: {rate_formatted}" if rate else ""
        else:
            # For other units (files, items, animals, etc.), show as-is
            laps_str = f":material/laps: {int(n)} {unit} / {int(total)} {unit}"
            rate_str = f":material/speed: {rate:.1f} {unit}/s" if rate else ""
        
        elapsed_str = f":material/timer: {fmt_time(elapsed)}" if elapsed else ""
        eta_str = f":material/sports_score: {fmt_time((total - n) / rate)}" if rate and total > n else ""
        
        label = self.label_divider + self.label_divider.join(filter(None, [
            percent_str, laps_str, rate_str, elapsed_str, eta_str
        ]))
        
        # Update the progress bar
        self.update(pbar_id, n - self.states[pbar_id], text=label)

