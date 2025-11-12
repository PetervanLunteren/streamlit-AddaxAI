"""
Lightweight keyboard shortcut helpers for Streamlit.

Implements the same mapping behavior as `streamlit-shortcuts` but hides
the injected iframe so no extra whitespace appears in the layout.
"""

from __future__ import annotations

import json
import streamlit.components.v1 as components


def _hidden_component(html: str) -> None:
    """Render hidden HTML (no visual footprint)."""
    components.html(
        f"""
        <div style="margin:0;padding:0;">
        <script>
        const frame = window.frameElement;
        if (frame) {{
            frame.style.position = "absolute";
            frame.style.width = "0px";
            frame.style.height = "0px";
            frame.style.border = "0";
            frame.style.opacity = "0";
            frame.style.pointerEvents = "none";
        }}
        </script>
        {html}
        </div>
        """,
        height=0,
        width=0,
    )


def register_shortcuts(**shortcuts: str | list[str]) -> None:
    """Register keyboard shortcuts for the given Streamlit element keys."""
    if not shortcuts:
        return

    normalized = {}
    for key, value in shortcuts.items():
        normalized[key] = [value] if isinstance(value, str) else list(value)

    js = f"""
    <script>
    (function() {{
        const doc = window.parent.document;
        const parentWindow = window.parent;
        const shortcuts = {json.dumps(normalized)};

        if (!parentWindow.__AddaxShortcutListener) {{
            parentWindow.__AddaxShortcutMap = {{}};
            parentWindow.__AddaxShortcutListener = function(e) {{
                const allShortcuts = parentWindow.__AddaxShortcutMap || {{}};
                for (const [key, shortcutList] of Object.entries(allShortcuts)) {{
                    for (const shortcut of shortcutList) {{
                        const parts = shortcut.toLowerCase().split('+');
                        const hasCtrl = parts.includes('ctrl');
                        const hasAlt = parts.includes('alt');
                        const hasShift = parts.includes('shift');
                        const hasMeta = parts.includes('meta') || parts.includes('cmd');
                        const mainKey = parts.find(
                            p => !['ctrl','alt','shift','meta','cmd'].includes(p)
                        );

                        if (
                            hasCtrl === e.ctrlKey &&
                            hasAlt === e.altKey &&
                            hasShift === e.shiftKey &&
                            hasMeta === e.metaKey &&
                            e.key.toLowerCase() === mainKey
                        ) {{
                            e.preventDefault();
                            const selectors = [
                                `.st-key-${{key}} button`,
                                `.st-key-${{key}} input`,
                                `[data-testid=\"${{key}}\"]`,
                                `button:has([data-testid=\"baseButton-${{key}}\"])`,
                                `[aria-label=\"${{key}}\"]`
                            ];
                            let el = null;
                            for (const selector of selectors) {{
                                el = doc.querySelector(selector);
                                if (el) break;
                            }}
                            if (el) {{
                                el.click();
                                el.focus();
                            }}
                            return;
                        }}
                    }}
                }}
            }};
            doc.addEventListener('keydown', parentWindow.__AddaxShortcutListener);
        }}

        Object.assign(parentWindow.__AddaxShortcutMap, shortcuts);
    }})();
    </script>
    """
    _hidden_component(js)


def clear_shortcut_listeners() -> None:
    """Remove registered shortcuts and their listeners."""
    js = """
    <script>
    (function() {
        const doc = window.parent.document;
        const parentWindow = window.parent;
        if (parentWindow.__AddaxShortcutListener) {
            doc.removeEventListener('keydown', parentWindow.__AddaxShortcutListener);
            parentWindow.__AddaxShortcutListener = null;
        }
        parentWindow.__AddaxShortcutMap = {};
    })();
    </script>
    """
    _hidden_component(js)
