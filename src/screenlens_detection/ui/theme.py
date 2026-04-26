from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Theme:
    """Theme base.
    """

    # Window
    window_bg: Optional[str] = "#cbd4e6"

    # Panels (fallback for 1/2/3/5 if section-specific colors aren't set)
    panel_bg: Optional[str] = "#1d242d"
    panel_fg: Optional[str] = "#1d242d"

    # Controls panel
    controls_panel_bg: Optional[str] = "#1d242d"
    controls_panel_fg: Optional[str] = "#fcfdff"

    # Pipeline panel
    pipeline_panel_bg: Optional[str] = "#1d242d"
    pipeline_panel_fg: Optional[str] = "#fcfdff"

    # Runtime panel
    runtime_panel_bg: Optional[str] = "#1d242d"
    runtime_panel_fg: Optional[str] = "#fcfdff"

    # Detected panel
    detected_panel_bg: Optional[str] = "#3d4c5e"
    detected_panel_fg: Optional[str] = "#090b0e"

    # Inputs / buttons
    control_bg: Optional[str] = "#546881"
    control_fg: Optional[str] = "#090b0e"
    control_border: Optional[str] = "#090b0e"

    # Preview image labels
    preview_bg: Optional[str] = "#151a20"
    preview_fg: Optional[str] = "#fcfdff"

    radius_px: int = 8


def pick_section_bg(theme: Theme, *, section_bg: Optional[str]) -> Optional[str]:
    return section_bg if section_bg is not None else theme.panel_bg


def pick_section_fg(theme: Theme, *, section_fg: Optional[str]) -> Optional[str]:
    return section_fg if section_fg is not None else theme.panel_fg


def groupbox_style(*, bg: str, fg: str, radius_px: int) -> str:
    return (
        f"QGroupBox {{ background-color: {bg}; color: {fg}; border-radius: {radius_px}px; }}"
        f"QGroupBox QLabel, QGroupBox QCheckBox {{ color: {fg}; background-color: transparent; }}"
    )


def output_text_style(*, bg: str, fg: str) -> str:
    return f"QPlainTextEdit {{ background-color: {bg}; color: {fg}; border: none; }}"


def combobox_style(*, bg: str, fg: str, border: str, radius_px: int) -> str:
    return (
        "QComboBox { "
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: {radius_px}px; padding: 4px 28px 4px 8px; "
        "}"
        "QComboBox::drop-down { "
        "subcontrol-origin: padding; subcontrol-position: center right; "
        "width: 20px; background: transparent; border: none; margin: 0px; "
        "}"
    )


def spinbox_style(*, widget: str, bg: str, fg: str, border: str, radius_px: int) -> str:
    return (
        f"{widget} {{ "
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: {radius_px}px; padding: 4px 44px 4px 8px; "
        "}"
        f"{widget}::up-button {{ "
        "subcontrol-origin: padding; subcontrol-position: center right; "
        "width: 20px; background: transparent; border: none; right: 20px; "
        "}"
        f"{widget}::down-button {{ "
        "subcontrol-origin: padding; subcontrol-position: center right; "
        "width: 20px; background: transparent; border: none; right: 0px; "
        "}"
    )


def button_style(*, bg: str, fg: str, border: str, radius_px: int) -> str:
    return (
        "QPushButton { "
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: {radius_px}px; padding: 6px 12px; "
        "}"
        "QPushButton:hover { background-color: #5f7896; }"
        "QPushButton:pressed { background-color: #4a5c73; }"
    )
