# -*- coding: utf-8 -*-
"""
Design System Generator - Aggregates search results and applies reasoning
to generate comprehensive design system recommendations.

Usage:
    from design_system import generate_design_system
    result = generate_design_system("SaaS dashboard", "My Project")
    print(result["text"])

    # With persistence (Master + Overrides pattern)
    result = generate_design_system("SaaS dashboard", "My Project", persist=True, output_dir="/path/to/project")
    result["persistence"]  # {"status": "success"|"skipped_exists", "created_files": [...], ...}
    result = generate_design_system("SaaS dashboard", "My Project", persist=True, page="dashboard", output_dir="/path/to/project")
"""

import csv
import json
import os
import re
import sys
import io
from datetime import datetime
from pathlib import Path
from core import search, DATA_DIR

# Force UTF-8 for stdout/stderr to handle emojis/box-drawing chars on Windows (cp1252 default)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============ CONFIGURATION ============
REASONING_FILE = "ui-reasoning.csv"

SEARCH_CONFIG = {
    "product": {"max_results": 1},
    "style": {"max_results": 3},
    "color": {"max_results": 2},
    "landing": {"max_results": 2},
    "typography": {"max_results": 2}
}

DIAL_TIERS = {
    "variance": [
        (1, 3, {"label": "Centered / Minimal", "style_keywords": ["Minimalism", "Exaggerated Minimalism", "centered", "symmetric", "grid-based"]}),
        (4, 7, {"label": "Balanced / Modern", "style_keywords": ["modern", "structured", "balanced"]}),
        (8, 10, {"label": "Bold / Asymmetric", "style_keywords": ["Brutalism", "Bento Grids", "asymmetric", "experimental"]}),
    ],
    "motion": [
        (1, 3, {"label": "Subtle", "tier": "Subtle"}),
        (4, 7, {"label": "Standard", "tier": "Standard"}),
        (8, 10, {"label": "Complex", "tier": "Complex"}),
    ],
    "density": [
        (1, 3, {"label": "Spacious", "spacing": {"xs": "4px", "sm": "8px", "md": "24px", "lg": "32px", "xl": "48px", "2xl": "64px", "3xl": "96px"}}),
        (4, 7, {"label": "Standard", "spacing": {"xs": "4px", "sm": "8px", "md": "16px", "lg": "24px", "xl": "32px", "2xl": "48px", "3xl": "64px"}}),
        (8, 10, {"label": "Dense / Dashboard", "spacing": {"xs": "2px", "sm": "4px", "md": "8px", "lg": "12px", "xl": "16px", "2xl": "24px", "3xl": "32px"}}),
    ],
}


def _resolve_dial(dial_name: str, value) -> dict:
    if value is None:
        return None
    value = max(1, min(10, int(value)))
    for lo, hi, info in DIAL_TIERS[dial_name]:
        if lo <= value <= hi:
            return {**info, "value": value}
    return None

_DARK_PRIMARY_MARKERS = (
    "dark mode primary", "dark primary", "dark-only", "dark only",
    "dark preferred", "dark focused", "dark-first", "dark rich",
    "light mode only as exception",
)

_DARK_QUERY_MARKERS = (
    "dark mode", "dark theme", "dark ui", "dark-mode", "darkmode",
    "night mode", "midnight", "oled",
)

_DARK_ANTI_PATTERN_MARKERS = ("dark mode", "dark modes", "dark theme")

_DARK_BACKGROUND_MAX_LUMINANCE = 0.18


def _relative_luminance(hex_color: str):
    if not hex_color:
        return None
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        return None
    try:
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _palette_is_dark(palette: dict) -> bool:
    luminance = _relative_luminance((palette or {}).get("Background", ""))
    return luminance is not None and luminance < _DARK_BACKGROUND_MAX_LUMINANCE


def _style_is_dark_primary(style: dict) -> bool:
    if not style:
        return False
    declared = "{} {}".format(
        style.get("Light Mode ✓", ""), style.get("Dark Mode ✓", "")
    ).lower()
    return any(marker in declared for marker in _DARK_PRIMARY_MARKERS)


def _query_wants_dark(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in _DARK_QUERY_MARKERS)


def _resolve_color_mode(query: str, style: dict) -> str:
    if _query_wants_dark(query) or _style_is_dark_primary(style):
        return "dark"
    return "light"


def _select_palette_for_mode(palettes: list, mode: str) -> dict:
    if not palettes:
        return {}
    if mode == "dark":
        for palette in palettes:
            if _palette_is_dark(palette):
                return palette
    return palettes[0]


def _filter_anti_patterns_for_mode(anti_patterns: str, mode: str) -> str:
    if mode != "dark" or not anti_patterns:
        return anti_patterns
    kept = [
        clause for clause in anti_patterns.split("+")
        if not any(marker in clause.lower() for marker in _DARK_ANTI_PATTERN_MARKERS)
    ]
    return " + ".join(clause.strip() for clause in kept if clause.strip())

class DesignSystemGenerator:
    def __init__(self):
        self.reasoning_data = self._load_reasoning()

    def _load_reasoning(self) -> list:
        filepath = DATA_DIR / REASONING_FILE
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def _multi_domain_search(self, query: str, style_priority: list = None) -> dict:
        results = {}
        for domain, config in SEARCH_CONFIG.items():
            if domain == "style" and style_priority:
                priority_query = " ".join(style_priority[:2]) if style_priority else query
                combined_query = f"{query} {priority_query}"
                results[domain] = search(combined_query, domain, config["max_results"])
            else:
                results[domain] = search(query, domain, config["max_results"])
        return results

    def _find_reasoning_rule(self, category: str) -> dict:
        category_lower = category.lower()

        for rule in self.reasoning_data:
            if rule.get("UI_Category", "").lower() == category_lower:
                return rule

        for rule in self.reasoning_data:
            ui_cat = rule.get("UI_Category", "").lower()
            if ui_cat in category_lower or category_lower in ui_cat:
                return rule

        for rule in self.reasoning_data:
            ui_cat = rule.get("UI_Category", "").lower()
            keywords = ui_cat.replace("/", " ").replace("-", " ").split()
            if any(kw in category_lower for kw in keywords):
                return rule

        return {}

    def _apply_reasoning(self, category: str, search_results: dict) -> dict:
        rule = self._find_reasoning_rule(category)

        if not rule:
            return {
                "pattern": "Hero + Features + CTA",
                "style_priority": ["Minimalism", "Flat Design"],
                "color_mood": "Professional",
                "typography_mood": "Clean",
                "key_effects": "Subtle hover transitions",
                "anti_patterns": "",
                "decision_rules": {},
                "severity": "MEDIUM"
            }

        decision_rules = {}
        try:
            decision_rules = json.loads(rule.get("Decision_Rules", "{}"))
        except json.JSONDecodeError:
            pass

        return {
            "pattern": rule.get("Recommended_Pattern", ""),
            "style_priority": [s.strip() for s in rule.get("Style_Priority", "").split("+")],
            "color_mood": rule.get("Color_Mood", ""),
            "typography_mood": rule.get("Typography_Mood", ""),
            "key_effects": rule.get("Key_Effects", ""),
            "anti_patterns": rule.get("Anti_Patterns", ""),
            "decision_rules": decision_rules,
            "severity": rule.get("Severity", "MEDIUM")
        }

    def _select_best_match(self, results: list, priority_keywords: list) -> dict:
        if not results:
            return {}

        if not priority_keywords:
            return results[0]

        for priority in priority_keywords:
            priority_lower = priority.lower().strip()
            for result in results:
                style_name = result.get("Style Category", "").lower()
                if priority_lower in style_name or style_name in priority_lower:
                    return result

        scored = []
        for result in results:
            result_str = str(result).lower()
            score = 0
            for kw in priority_keywords:
                kw_lower = kw.lower().strip()
                if kw_lower in result.get("Style Category", "").lower():
                    score += 10
                elif kw_lower in result.get("Keywords", "").lower():
                    score += 3
                elif kw_lower in result_str:
                    score += 1
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else results[0]

    def _extract_results(self, search_result: dict) -> list:
        return search_result.get("results", [])

    def generate(self, query: str, project_name: str = None,
                 variance: int = None, motion: int = None, density: int = None) -> dict:
        variance_info = _resolve_dial("variance", variance)
        motion_info = _resolve_dial("motion", motion)
        density_info = _resolve_dial("density", density)

        product_result = search(query, "product", 1)
        product_results = product_result.get("results", [])
        category = "General"
        if product_results:
            category = product_results[0].get("Product Type", "General")

        reasoning = self._apply_reasoning(category, {})
        style_priority = reasoning.get("style_priority", [])

        effective_style_priority = style_priority
        if variance_info:
            effective_style_priority = variance_info["style_keywords"] + style_priority

        search_results = self._multi_domain_search(query, effective_style_priority)
        search_results["product"] = product_result

        style_results = self._extract_results(search_results.get("style", {}))
        color_results = self._extract_results(search_results.get("color", {}))
        typography_results = self._extract_results(search_results.get("typography", {}))
        landing_results = self._extract_results(search_results.get("landing", {}))

        best_style = self._select_best_match(style_results, effective_style_priority)
        color_mode = _resolve_color_mode(query, best_style)
        best_color = _select_palette_for_mode(color_results, color_mode)
        best_typography = typography_results[0] if typography_results else {}
        best_landing = landing_results[0] if landing_results else {}

        motion_snippet = {}
        if motion_info:
            motion_result = search(f"{query} {motion_info['tier']}", "gsap", 5)
            motion_matches = motion_result.get("results", [])
            tiered = [m for m in motion_matches if m.get("Intensity Tier") == motion_info["tier"]]
            if tiered:
                motion_snippet = tiered[0]
            elif motion_matches:
                motion_snippet = motion_matches[0]

        style_effects = best_style.get("Effects & Animation", "")
        reasoning_effects = reasoning.get("key_effects", "")
        combined_effects = style_effects if style_effects else reasoning_effects

        return {
            "project_name": project_name or query.upper(),
            "category": category,
            "pattern": {
                "name": best_landing.get("Pattern Name", reasoning.get("pattern", "Hero + Features + CTA")),
                "sections": best_landing.get("Section Order", "Hero > Features > CTA"),
                "cta_placement": best_landing.get("Primary CTA Placement", "Above fold"),
                "color_strategy": best_landing.get("Color Strategy", ""),
                "conversion": best_landing.get("Conversion Optimization", "")
            },
            "style": {
                "name": best_style.get("Style Category", "Minimalism"),
                "type": best_style.get("Type", "General"),
                "effects": style_effects,
                "keywords": best_style.get("Keywords", ""),
                "best_for": best_style.get("Best For", ""),
                "performance": best_style.get("Performance", ""),
                "accessibility": best_style.get("Accessibility", ""),
                "light_mode": best_style.get("Light Mode ✓", ""),
                "dark_mode": best_style.get("Dark Mode ✓", ""),
            },
            "colors": {
                "primary": best_color.get("Primary", "#2563EB"),
                "on_primary": best_color.get("On Primary", ""),
                "secondary": best_color.get("Secondary", "#3B82F6"),
                "accent": best_color.get("Accent", "#F97316"),
                "background": best_color.get("Background", "#F8FAFC"),
                "foreground": best_color.get("Foreground", "#1E293B"),
                "muted": best_color.get("Muted", ""),
                "border": best_color.get("Border", ""),
                "destructive": best_color.get("Destructive", ""),
                "ring": best_color.get("Ring", ""),
                "notes": best_color.get("Notes", ""),
                "cta": best_color.get("Accent", "#F97316"),
                "text": best_color.get("Foreground", "#1E293B"),
            },
            "typography": {
                "heading": best_typography.get("Heading Font", "Inter"),
                "body": best_typography.get("Body Font", "Inter"),
                "mood": best_typography.get("Mood/Style Keywords", reasoning.get("typography_mood", "")),
                "best_for": best_typography.get("Best For", ""),
                "google_fonts_url": best_typography.get("Google Fonts URL", ""),
                "css_import": best_typography.get("CSS Import", "")
            },
            "key_effects": combined_effects,
            "anti_patterns": _filter_anti_patterns_for_mode(
                reasoning.get("anti_patterns", ""), color_mode
            ),
            "decision_rules": reasoning.get("decision_rules", {}),
            "severity": reasoning.get("severity", "MEDIUM"),
            "dials": {
                "variance": variance_info["value"] if variance_info else None,
                "variance_label": variance_info["label"] if variance_info else None,
                "motion": motion_info["value"] if motion_info else None,
                "motion_label": motion_info["label"] if motion_info else None,
                "density": density_info["value"] if density_info else None,
                "density_label": density_info["label"] if density_info else None,
            },
            "motion_snippet": motion_snippet,
            "spacing_scale": density_info["spacing"] if density_info else None,
        }

BOX_WIDTH = 90


def hex_to_ansi(hex_color: str) -> str:
    if not hex_color or not hex_color.startswith('#'):
        return ""
    colorterm = os.environ.get('COLORTERM', '')
    if colorterm not in ('truecolor', '24bit'):
        return ""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return ""
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m██\033[0m "


def ansi_ljust(s: str, width: int) -> str:
    import re
    visible_len = len(re.sub(r'\033\[[0-9;]*m', '', s))
    pad = width - visible_len
    return s + (" " * max(0, pad))


def section_header(name: str, width: int) -> str:
    label = f"─── {name} "
    fill = "─" * (width - len(label) - 1)
    return f"├{label}{fill}┤"


def format_ascii_box(design_system: dict) -> str:
    project = design_system.get("project_name", "PROJECT")
    pattern = design_system.get("pattern", {})
    style = design_system.get("style", {})
    colors = design_system.get("colors", {})
    typography = design_system.get("typography", {})
    effects = design_system.get("key_effects", "")
    anti_patterns = design_system.get("anti_patterns", "")
    dials = design_system.get("dials", {})
    motion_snippet = design_system.get("motion_snippet", {})

    def wrap_text(text: str, prefix: str, width: int) -> list:
        if not text:
            return []
        words = text.split()
        lines = []
        current_line = prefix
        for word in words:
            if len(current_line) + len(word) + 1 <= width - 2:
                current_line += (" " if current_line != prefix else "") + word
            else:
                if current_line != prefix:
                    lines.append(current_line)
                current_line = prefix + word
        if current_line != prefix:
            lines.append(current_line)
        return lines

    sections = pattern.get("sections", "").split(">")
    sections = [s.strip() for s in sections if s.strip()]

    lines = []
    w = BOX_WIDTH - 1

    lines.append("╔" + "═" * w + "╗")
    lines.append(ansi_ljust(f"║  TARGET: {project} - RECOMMENDED DESIGN SYSTEM", BOX_WIDTH) + "║")
    lines.append("╚" + "═" * w + "╝")
    lines.append("┌" + "─" * w + "┐")

    if any(dials.get(k) is not None for k in ("variance", "motion", "density")):
        lines.append(section_header("DESIGN DIALS", BOX_WIDTH + 1))
        if dials.get("variance") is not None:
            lines.append(f"│  Variance: {dials['variance']}/10 — {dials['variance_label']}".ljust(BOX_WIDTH) + "│")
        if dials.get("motion") is not None:
            lines.append(f"│  Motion:   {dials['motion']}/10 — {dials['motion_label']}".ljust(BOX_WIDTH) + "│")
        if dials.get("density") is not None:
            lines.append(f"│  Density:  {dials['density']}/10 — {dials['density_label']}".ljust(BOX_WIDTH) + "│")

    lines.append(section_header("PATTERN", BOX_WIDTH + 1))
    lines.append(f"│  Name: {pattern.get('name', '')}".ljust(BOX_WIDTH) + "│")
    if pattern.get('conversion'):
        lines.append(f"│     Conversion: {pattern.get('conversion', '')}".ljust(BOX_WIDTH) + "│")
    if pattern.get('cta_placement'):
        lines.append(f"│     CTA: {pattern.get('cta_placement', '')}".ljust(BOX_WIDTH) + "│")
    lines.append("│     Sections:".ljust(BOX_WIDTH) + "│")
    for i, section in enumerate(sections, 1):
        lines.append(f"│       {i}. {section}".ljust(BOX_WIDTH) + "│")

    lines.append(section_header("STYLE", BOX_WIDTH + 1))
    lines.append(f"│  Name: {style.get('name', '')}".ljust(BOX_WIDTH) + "│")
    light = style.get("light_mode", "")
    dark = style.get("dark_mode", "")
    if light or dark:
        lines.append(f"│     Mode Support: Light {light}  Dark {dark}".ljust(BOX_WIDTH) + "│")
    if style.get("keywords"):
        for line in wrap_text(f"Keywords: {style.get('keywords', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")
    if style.get("best_for"):
        for line in wrap_text(f"Best For: {style.get('best_for', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")
    if style.get("performance") or style.get("accessibility"):
        perf_a11y = f"Performance: {style.get('performance', '')} | Accessibility: {style.get('accessibility', '')}"
        lines.append(f"│     {perf_a11y}".ljust(BOX_WIDTH) + "│")

    lines.append(section_header("COLORS", BOX_WIDTH + 1))
    color_entries = [
        ("Primary",      "primary",      "--color-primary"),
        ("On Primary",   "on_primary",   "--color-on-primary"),
        ("Secondary",    "secondary",    "--color-secondary"),
        ("Accent/CTA",   "accent",       "--color-accent"),
        ("Background",   "background",   "--color-background"),
        ("Foreground",   "foreground",   "--color-foreground"),
        ("Muted",        "muted",        "--color-muted"),
        ("Border",       "border",       "--color-border"),
        ("Destructive",  "destructive",  "--color-destructive"),
        ("Ring",         "ring",         "--color-ring"),
    ]
    for label, key, css_var in color_entries:
        hex_val = colors.get(key, "")
        if not hex_val:
            continue
        swatch = hex_to_ansi(hex_val)
        content = f"│     {swatch}{label + ':':14s} {hex_val:10s} ({css_var})"
        lines.append(ansi_ljust(content, BOX_WIDTH) + "│")
    if colors.get("notes"):
        for line in wrap_text(f"Notes: {colors.get('notes', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")

    lines.append(section_header("TYPOGRAPHY", BOX_WIDTH + 1))
    lines.append(f"│  {typography.get('heading', '')} / {typography.get('body', '')}".ljust(BOX_WIDTH) + "│")
    if typography.get("mood"):
        for line in wrap_text(f"Mood: {typography.get('mood', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")
    if typography.get("best_for"):
        for line in wrap_text(f"Best For: {typography.get('best_for', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")
    if typography.get("google_fonts_url"):
        lines.append(f"│     Google Fonts: {typography.get('google_fonts_url', '')}".ljust(BOX_WIDTH) + "│")
    if typography.get("css_import"):
        lines.append(f"│     CSS Import: {typography.get('css_import', '')[:70]}...".ljust(BOX_WIDTH) + "│")

    if effects:
        lines.append(section_header("KEY EFFECTS", BOX_WIDTH + 1))
        for line in wrap_text(effects, "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")

    if motion_snippet:
        lines.append(section_header("MOTION", BOX_WIDTH + 1))
        lines.append(f"│  {motion_snippet.get('Category', '')} ({motion_snippet.get('Intensity Tier', '')})".ljust(BOX_WIDTH) + "│")
        lines.append(f"│     Trigger: {motion_snippet.get('Trigger', '')} | Duration: {motion_snippet.get('Duration', '')} | Easing: {motion_snippet.get('Easing', '')}".ljust(BOX_WIDTH) + "│")
        for line in wrap_text(f"GSAP: {motion_snippet.get('GSAP Snippet', '')}", "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")
        if motion_snippet.get("Framework Notes"):
            for line in wrap_text(f"Framework: {motion_snippet.get('Framework Notes', '')}", "│     ", BOX_WIDTH):
                lines.append(line.ljust(BOX_WIDTH) + "│")

    if anti_patterns:
        lines.append(section_header("AVOID", BOX_WIDTH + 1))
        for line in wrap_text(anti_patterns, "│     ", BOX_WIDTH):
            lines.append(line.ljust(BOX_WIDTH) + "│")

    lines.append(section_header("PRE-DELIVERY CHECKLIST", BOX_WIDTH + 1))
    checklist_items = [
        "[ ] No emojis as icons (use SVG: Heroicons/Lucide)",
        "[ ] cursor-pointer on all clickable elements",
        "[ ] Hover states with smooth transitions (150-300ms)",
        "[ ] Light mode: text contrast 4.5:1 minimum",
        "[ ] Focus states visible for keyboard nav",
        "[ ] prefers-reduced-motion respected",
        "[ ] Responsive: 375px, 768px, 1024px, 1440px"
    ]
    for item in checklist_items:
        lines.append(f"│     {item}".ljust(BOX_WIDTH) + "│")

    lines.append("└" + "─" * w + "┘")

    return "\n".join(lines)


def format_markdown(design_system: dict) -> str:
    project = design_system.get("project_name", "PROJECT")
    pattern = design_system.get("pattern", {})
    style = design_system.get("style", {})
    colors = design_system.get("colors", {})
    typography = design_system.get("typography", {})
    effects = design_system.get("key_effects", "")
    anti_patterns = design_system.get("anti_patterns", "")
    dials = design_system.get("dials", {})
    motion_snippet = design_system.get("motion_snippet", {})

    lines = []
    lines.append(f"## Design System: {project}")
    lines.append("")

    if any(dials.get(k) is not None for k in ("variance", "motion", "density")):
        lines.append("### Design Dials")
        if dials.get("variance") is not None:
            lines.append(f"- **Variance:** {dials['variance']}/10 — {dials['variance_label']}")
        if dials.get("motion") is not None:
            lines.append(f"- **Motion:** {dials['motion']}/10 — {dials['motion_label']}")
        if dials.get("density") is not None:
            lines.append(f"- **Density:** {dials['density']}/10 — {dials['density_label']}")
        lines.append("")

    lines.append("### Pattern")
    lines.append(f"- **Name:** {pattern.get('name', '')}")
    if pattern.get('conversion'):
        lines.append(f"- **Conversion Focus:** {pattern.get('conversion', '')}")
    if pattern.get('cta_placement'):
        lines.append(f"- **CTA Placement:** {pattern.get('cta_placement', '')}")
    if pattern.get('color_strategy'):
        lines.append(f"- **Color Strategy:** {pattern.get('color_strategy', '')}")
    lines.append(f"- **Sections:** {pattern.get('sections', '')}")
    lines.append("")

    lines.append("### Style")
    lines.append(f"- **Name:** {style.get('name', '')}")
    light = style.get("light_mode", "")
    dark = style.get("dark_mode", "")
    if light or dark:
        lines.append(f"- **Mode Support:** Light {light} | Dark {dark}")
    if style.get('keywords'):
        lines.append(f"- **Keywords:** {style.get('keywords', '')}")
    if style.get('best_for'):
        lines.append(f"- **Best For:** {style.get('best_for', '')}")
    if style.get('performance') or style.get('accessibility'):
        lines.append(f"- **Performance:** {style.get('performance', '')} | **Accessibility:** {style.get('accessibility', '')}")
    lines.append("")

    lines.append("### Colors")
    lines.append("| Role | Hex | CSS Variable |")
    lines.append("|------|-----|--------------|")
    md_color_entries = [
        ("Primary",      "primary",      "--color-primary"),
        ("On Primary",   "on_primary",   "--color-on-primary"),
        ("Secondary",    "secondary",    "--color-secondary"),
        ("Accent/CTA",   "accent",       "--color-accent"),
        ("Background",   "background",   "--color-background"),
        ("Foreground",   "foreground",   "--color-foreground"),
        ("Muted",        "muted",        "--color-muted"),
        ("Border",       "border",       "--color-border"),
        ("Destructive",  "destructive",  "--color-destructive"),
        ("Ring",         "ring",         "--color-ring"),
    ]
    for label, key, css_var in md_color_entries:
        hex_val = colors.get(key, "")
        if hex_val:
            lines.append(f"| {label} | `{hex_val}` | `{css_var}` |")
    if colors.get("notes"):
        lines.append(f"\n*Notes: {colors.get('notes', '')}*")
    lines.append("")

    lines.append("### Typography")
    lines.append(f"- **Heading:** {typography.get('heading', '')}")
    lines.append(f"- **Body:** {typography.get('body', '')}")
    if typography.get("mood"):
        lines.append(f"- **Mood:** {typography.get('mood', '')}")
    if typography.get("best_for"):
        lines.append(f"- **Best For:** {typography.get('best_for', '')}")
    if typography.get("google_fonts_url"):
        lines.append(f"- **Google Fonts:** {typography.get('google_fonts_url', '')}")

    if typography.get("css_import"):
        lines.append(f"- **CSS Import:**")
        lines.append(f"```css")
        lines.append(f"{typography.get('css_import', '')}")
        lines.append(f"```")
    lines.append("")

    if effects:
        lines.append("### Key Effects")
        lines.append(f"{effects}")
        lines.append("")

    if motion_snippet:
        lines.append("### Motion")
        lines.append(f"**{motion_snippet.get('Category', '')}** ({motion_snippet.get('Intensity Tier', '')}) — Trigger: {motion_snippet.get('Trigger', '')} | Duration: {motion_snippet.get('Duration', '')} | Easing: `{motion_snippet.get('Easing', '')}`")
        lines.append("```js")
        lines.append(motion_snippet.get("GSAP Snippet", ""))
        lines.append("```")
        if motion_snippet.get("Framework Notes"):
            lines.append(f"*Framework notes: {motion_snippet.get('Framework Notes', '')}*")
        motion_do = motion_snippet.get("Do", "")
        motion_dont = motion_snippet.get("Don't", "")
        if motion_do:
            lines.append(f"- ✅ {motion_do}")
        if motion_dont:
            lines.append(f"- ❌ {motion_dont}")
        lines.append("")

    if anti_patterns:
        lines.append("### Avoid (Anti-patterns)")
        newline_bullet = '\n- '
        lines.append(f"- {anti_patterns.replace(' + ', newline_bullet)}")
        lines.append("")

    lines.append("### Pre-Delivery Checklist")
    lines.append("- [ ] No emojis as icons (use SVG: Heroicons/Lucide)")
    lines.append("- [ ] cursor-pointer on all clickable elements")
    lines.append("- [ ] Hover states with smooth transitions (150-300ms)")
    lines.append("- [ ] Light mode: text contrast 4.5:1 minimum")
    lines.append("- [ ] Focus states visible for keyboard nav")
    lines.append("- [ ] prefers-reduced-motion respected")
    lines.append("- [ ] Responsive: 375px, 768px, 1024px, 1440px")
    lines.append("")

    return "\n".join(lines)


def generate_design_system(query: str, project_name: str = None, output_format: str = "ascii",
                           persist: bool = False, page: str = None, output_dir: str = None,
                           variance: int = None, motion: int = None, density: int = None,
                           force: bool = False) -> dict:
    generator = DesignSystemGenerator()
    design_system = generator.generate(query, project_name, variance=variance, motion=motion, density=density)

    persistence_result = None
    if persist:
        persistence_result = persist_design_system(design_system, page, output_dir, query, force=force)

    text = format_markdown(design_system) if output_format == "markdown" else format_ascii_box(design_system)

    return {
        "text": text,
        "design_system": design_system,
        "persistence": persistence_result,
    }

def safe_slug(name, fallback: str = "default") -> str:
    slug = re.sub(r'[^a-z0-9_-]+', '-', str(name).lower()).strip('-')
    return slug or fallback


def persist_design_system(design_system: dict, page: str = None, output_dir: str = None,
                           page_query: str = None, force: bool = False) -> dict:
    base_dir = Path(output_dir) if output_dir else Path.cwd()

    project_name = design_system.get("project_name") or "default"
    project_slug = safe_slug(project_name)

    design_system_dir = base_dir / "design-system" / project_slug
    pages_dir = design_system_dir / "pages"

    master_file = design_system_dir / "MASTER.md"

    if master_file.exists() and not force:
        return {
            "status": "skipped_exists",
            "design_system_dir": str(design_system_dir),
            "master_file": str(master_file),
            "created_files": [],
            "message": (
                f"{master_file} already exists and was not modified. "
                "Read it first to check for prior design decisions, then "
                "re-run with force=True / --force to overwrite."
            ),
        }

    created_files = []

    design_system_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    master_content = format_master_md(design_system)
    with open(master_file, 'w', encoding='utf-8') as f:
        f.write(master_content)
    created_files.append(str(master_file))

    if page:
        page_file = pages_dir / f"{safe_slug(page, 'page')}.md"
        page_content = format_page_override_md(design_system, page, page_query)
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(page_content)
        created_files.append(str(page_file))

    return {
        "status": "success",
        "design_system_dir": str(design_system_dir),
        "master_file": str(master_file),
        "created_files": created_files
    }


def format_master_md(design_system: dict) -> str:
    project = design_system.get("project_name", "PROJECT")
    pattern = design_system.get("pattern", {})
    style = design_system.get("style", {})
    colors = design_system.get("colors", {})
    typography = design_system.get("typography", {})
    effects = design_system.get("key_effects", "")
    anti_patterns = design_system.get("anti_patterns", "")
    dials = design_system.get("dials", {})
    motion_snippet = design_system.get("motion_snippet", {})
    spacing_scale = design_system.get("spacing_scale")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    lines.append("# Design System Master File")
    lines.append("")
    lines.append("> **LOGIC:** When building a specific page, first check `design-system/pages/[page-name].md`.")
    lines.append("> If that file exists, its rules **override** this Master file.")
    lines.append("> If not, strictly follow the rules below.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(f"**Project:** {project}")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Category:** {design_system.get('category', 'General')}")
    if any(dials.get(k) is not None for k in ("variance", "motion", "density")):
        dial_parts = []
        if dials.get("variance") is not None:
            dial_parts.append(f"Variance {dials['variance']}/10 ({dials['variance_label']})")
        if dials.get("motion") is not None:
            dial_parts.append(f"Motion {dials['motion']}/10 ({dials['motion_label']})")
        if dials.get("density") is not None:
            dial_parts.append(f"Density {dials['density']}/10 ({dials['density_label']})")
        lines.append(f"**Design Dials:** {' | '.join(dial_parts)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Global Rules")
    lines.append("")

    lines.append("### Color Palette")
    lines.append("")
    lines.append("| Role | Hex | CSS Variable |")
    lines.append("|------|-----|--------------|")
    master_color_entries = [
        ("Primary",      "primary",      "--color-primary"),
        ("On Primary",   "on_primary",   "--color-on-primary"),
        ("Secondary",    "secondary",    "--color-secondary"),
        ("Accent/CTA",   "accent",       "--color-accent"),
        ("Background",   "background",   "--color-background"),
        ("Foreground",   "foreground",   "--color-foreground"),
        ("Muted",        "muted",        "--color-muted"),
        ("Border",       "border",       "--color-border"),
        ("Destructive",  "destructive",  "--color-destructive"),
        ("Ring",         "ring",         "--color-ring"),
    ]
    for label, key, css_var in master_color_entries:
        hex_val = colors.get(key, "")
        if hex_val:
            lines.append(f"| {label} | `{hex_val}` | `{css_var}` |")
    lines.append("")
    if colors.get("notes"):
        lines.append(f"**Color Notes:** {colors.get('notes', '')}")
        lines.append("")

    lines.append("### Typography")
    lines.append("")
    lines.append(f"- **Heading Font:** {typography.get('heading', 'Inter')}")
    lines.append(f"- **Body Font:** {typography.get('body', 'Inter')}")
    if typography.get("mood"):
        lines.append(f"- **Mood:** {typography.get('mood', '')}")
    if typography.get("google_fonts_url"):
        lines.append(f"- **Google Fonts:** [{typography.get('heading', '')} + {typography.get('body', '')}]({typography.get('google_fonts_url', '')})")
    lines.append("")
    if typography.get("css_import"):
        lines.append("**CSS Import:**")
        lines.append("```css")
        lines.append(typography.get("css_import", ""))
        lines.append("```")
        lines.append("")

    default_spacing = DIAL_TIERS["density"][1][2]["spacing"]
    scale = spacing_scale or default_spacing
    spacing_usage = {
        "xs": "Tight gaps", "sm": "Icon gaps, inline spacing", "md": "Standard padding",
        "lg": "Section padding", "xl": "Large gaps", "2xl": "Section margins", "3xl": "Hero padding",
    }
    lines.append("### Spacing Variables")
    lines.append("")
    if spacing_scale:
        lines.append(f"*Density: {dials.get('density')}/10 — {dials.get('density_label')}*")
        lines.append("")
    lines.append("| Token | Value | Usage |")
    lines.append("|-------|-------|-------|")
    for token in ("xs", "sm", "md", "lg", "xl", "2xl", "3xl"):
        px_value = scale[token]
        rem_value = f"{int(px_value.rstrip('px')) / 16:g}rem"
        lines.append(f"| `--space-{token}` | `{px_value}` / `{rem_value}` | {spacing_usage[token]} |")
    lines.append("")

    lines.append("### Shadow Depths")
    lines.append("")
    lines.append("| Level | Value | Usage |")
    lines.append("|-------|-------|-------|")
    lines.append("| `--shadow-sm` | `0 1px 2px rgba(0,0,0,0.05)` | Subtle lift |")
    lines.append("| `--shadow-md` | `0 4px 6px rgba(0,0,0,0.1)` | Cards, buttons |")
    lines.append("| `--shadow-lg` | `0 10px 15px rgba(0,0,0,0.1)` | Modals, dropdowns |")
    lines.append("| `--shadow-xl` | `0 20px 25px rgba(0,0,0,0.15)` | Hero images, featured cards |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Component Specs")
    lines.append("")

    lines.append("### Buttons")
    lines.append("")
    lines.append("```css")
    lines.append("/* Primary Button */")
    lines.append(".btn-primary {")
    lines.append(f"  background: {colors.get('cta', '#F97316')};")
    lines.append("  color: white;")
    lines.append("  padding: 12px 24px;")
    lines.append("  border-radius: 8px;")
    lines.append("  font-weight: 600;")
    lines.append("  transition: all 200ms ease;")
    lines.append("  cursor: pointer;")
    lines.append("}")
    lines.append("")
    lines.append(".btn-primary:hover {")
    lines.append("  opacity: 0.9;")
    lines.append("  transform: translateY(-1px);")
    lines.append("}")
    lines.append("")
    lines.append("/* Secondary Button */")
    lines.append(".btn-secondary {")
    lines.append(f"  background: transparent;")
    lines.append(f"  color: {colors.get('primary', '#2563EB')};")
    lines.append(f"  border: 2px solid {colors.get('primary', '#2563EB')};")
    lines.append("  padding: 12px 24px;")
    lines.append("  border-radius: 8px;")
    lines.append("  font-weight: 600;")
    lines.append("  transition: all 200ms ease;")
    lines.append("  cursor: pointer;")
    lines.append("}")
    lines.append("```")
    lines.append("")

    lines.append("### Cards")
    lines.append("")
    lines.append("```css")
    lines.append(".card {")
    lines.append(f"  background: {colors.get('background', '#FFFFFF')};")
    lines.append("  border-radius: 12px;")
    lines.append("  padding: 24px;")
    lines.append("  box-shadow: var(--shadow-md);")
    lines.append("  transition: all 200ms ease;")
    lines.append("  cursor: pointer;")
    lines.append("}")
    lines.append("")
    lines.append(".card:hover {")
    lines.append("  box-shadow: var(--shadow-lg);")
    lines.append("  transform: translateY(-2px);")
    lines.append("}")
    lines.append("```")
    lines.append("")

    lines.append("### Inputs")
    lines.append("")
    lines.append("```css")
    lines.append(".input {")
    lines.append("  padding: 12px 16px;")
    lines.append("  border: 1px solid #E2E8F0;")
    lines.append("  border-radius: 8px;")
    lines.append("  font-size: 16px;")
    lines.append("  transition: border-color 200ms ease;")
    lines.append("}")
    lines.append("")
    lines.append(".input:focus {")
    lines.append(f"  border-color: {colors.get('primary', '#2563EB')};")
    lines.append("  outline: none;")
    lines.append(f"  box-shadow: 0 0 0 3px {colors.get('primary', '#2563EB')}20;")
    lines.append("}")
    lines.append("```")
    lines.append("")

    lines.append("### Modals")
    lines.append("")
    lines.append("```css")
    lines.append(".modal-overlay {")
    lines.append("  background: rgba(0, 0, 0, 0.5);")
    lines.append("  backdrop-filter: blur(4px);")
    lines.append("}")
    lines.append("")
    lines.append(".modal {")
    lines.append("  background: white;")
    lines.append("  border-radius: 16px;")
    lines.append("  padding: 32px;")
    lines.append("  box-shadow: var(--shadow-xl);")
    lines.append("  max-width: 500px;")
    lines.append("  width: 90%;")
    lines.append("}")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Style Guidelines")
    lines.append("")
    lines.append(f"**Style:** {style.get('name', 'Minimalism')}")
    lines.append("")
    if style.get("keywords"):
        lines.append(f"**Keywords:** {style.get('keywords', '')}")
        lines.append("")
    if style.get("best_for"):
        lines.append(f"**Best For:** {style.get('best_for', '')}")
        lines.append("")
    if effects:
        lines.append(f"**Key Effects:** {effects}")
        lines.append("")

    lines.append("### Page Pattern")
    lines.append("")
    lines.append(f"**Pattern Name:** {pattern.get('name', '')}")
    lines.append("")
    if pattern.get('conversion'):
        lines.append(f"- **Conversion Strategy:** {pattern.get('conversion', '')}")
    if pattern.get('cta_placement'):
        lines.append(f"- **CTA Placement:** {pattern.get('cta_placement', '')}")
    lines.append(f"- **Section Order:** {pattern.get('sections', '')}")
    lines.append("")

    if motion_snippet:
        lines.append("---")
        lines.append("")
        lines.append("## Motion")
        lines.append("")
        lines.append(f"**{motion_snippet.get('Category', '')}** ({motion_snippet.get('Intensity Tier', '')}) — Trigger: {motion_snippet.get('Trigger', '')} | Duration: {motion_snippet.get('Duration', '')} | Easing: `{motion_snippet.get('Easing', '')}`")
        lines.append("")
        lines.append("```js")
        lines.append(motion_snippet.get("GSAP Snippet", ""))
        lines.append("```")
        lines.append("")
        if motion_snippet.get("Framework Notes"):
            lines.append(f"**Framework notes:** {motion_snippet.get('Framework Notes', '')}")
            lines.append("")
        motion_do = motion_snippet.get("Do", "")
        motion_dont = motion_snippet.get("Don't", "")
        if motion_do:
            lines.append(f"- ✅ {motion_do}")
        if motion_dont:
            lines.append(f"- ❌ {motion_dont}")
        if motion_snippet.get("Performance Notes"):
            lines.append(f"- ⚡ {motion_snippet.get('Performance Notes', '')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Anti-Patterns (Do NOT Use)")
    lines.append("")
    if anti_patterns:
        anti_list = [a.strip() for a in anti_patterns.split("+")]
        for anti in anti_list:
            if anti:
                lines.append(f"- ❌ {anti}")
    lines.append("")
    lines.append("### Additional Forbidden Patterns")
    lines.append("")
    lines.append("- ❌ **Emojis as icons** — Use SVG icons (Heroicons, Lucide, Simple Icons)")
    lines.append("- ❌ **Missing cursor:pointer** — All clickable elements must have cursor:pointer")
    lines.append("- ❌ **Layout-shifting hovers** — Avoid scale transforms that shift layout")
    lines.append("- ❌ **Low contrast text** — Maintain 4.5:1 minimum contrast ratio")
    lines.append("- ❌ **Instant state changes** — Always use transitions (150-300ms)")
    lines.append("- ❌ **Invisible focus states** — Focus states must be visible for a11y")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Pre-Delivery Checklist")
    lines.append("")
    lines.append("Before delivering any UI code, verify:")
    lines.append("")
    lines.append("- [ ] No emojis used as icons (use SVG instead)")
    lines.append("- [ ] All icons from consistent icon set (Heroicons/Lucide)")
    lines.append("- [ ] `cursor-pointer` on all clickable elements")
    lines.append("- [ ] Hover states with smooth transitions (150-300ms)")
    lines.append("- [ ] Light mode: text contrast 4.5:1 minimum")
    lines.append("- [ ] Focus states visible for keyboard navigation")
    lines.append("- [ ] `prefers-reduced-motion` respected")
    lines.append("- [ ] Responsive: 375px, 768px, 1024px, 1440px")
    lines.append("- [ ] No content hidden behind fixed navbars")
    lines.append("- [ ] No horizontal scroll on mobile")
    lines.append("")

    return "\n".join(lines)


def format_page_override_md(design_system: dict, page_name: str, page_query: str = None) -> str:
    project = design_system.get("project_name", "PROJECT")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    page_title = page_name.replace("-", " ").replace("_", " ").title()

    page_overrides = _generate_intelligent_overrides(page_name, page_query, design_system)

    lines = []

    lines.append(f"# {page_title} Page Overrides")
    lines.append("")
    lines.append(f"> **PROJECT:** {project}")
    lines.append(f"> **Generated:** {timestamp}")
    lines.append(f"> **Page Type:** {page_overrides.get('page_type', 'General')}")
    lines.append("")
    lines.append("> ⚠️ **IMPORTANT:** Rules in this file **override** the Master file (`design-system/MASTER.md`).")
    lines.append("> Only deviations from the Master are documented here. For all other rules, refer to the Master.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Page-Specific Rules")
    lines.append("")

    lines.append("### Layout Overrides")
    lines.append("")
    layout = page_overrides.get("layout", {})
    if layout:
        for key, value in layout.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("- No overrides — use Master layout")
    lines.append("")

    lines.append("### Spacing Overrides")
    lines.append("")
    spacing = page_overrides.get("spacing", {})
    if spacing:
        for key, value in spacing.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("- No overrides — use Master spacing")
    lines.append("")

    lines.append("### Typography Overrides")
    lines.append("")
    typography = page_overrides.get("typography", {})
    if typography:
        for key, value in typography.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("- No overrides — use Master typography")
    lines.append("")

    lines.append("### Color Overrides")
    lines.append("")
    colors = page_overrides.get("colors", {})
    if colors:
        for key, value in colors.items():
            lines.append(f"- **{key}:** {value}")
    else:
        lines.append("- No overrides — use Master colors")
    lines.append("")

    lines.append("### Component Overrides")
    lines.append("")
    components = page_overrides.get("components", [])
    if components:
        for comp in components:
            lines.append(f"- {comp}")
    else:
        lines.append("- No overrides — use Master component specs")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Page-Specific Components")
    lines.append("")
    unique_components = page_overrides.get("unique_components", [])
    if unique_components:
        for comp in unique_components:
            lines.append(f"- {comp}")
    else:
        lines.append("- No unique components for this page")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    recommendations = page_overrides.get("recommendations", [])
    if recommendations:
        for rec in recommendations:
            lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


def _generate_intelligent_overrides(page_name: str, page_query: str, design_system: dict) -> dict:
    from core import search

    page_lower = page_name.lower()
    query_lower = (page_query or "").lower()
    combined_context = f"{page_lower} {query_lower}"

    style_search = search(combined_context, "style", max_results=1)
    ux_search = search(combined_context, "ux", max_results=3)
    landing_search = search(combined_context, "landing", max_results=1)

    style_results = style_search.get("results", [])
    ux_results = ux_search.get("results", [])
    landing_results = landing_search.get("results", [])

    page_type = _detect_page_type(combined_context, style_results)

    layout = {}
    spacing = {}
    typography = {}
    colors = {}
    components = []
    unique_components = []
    recommendations = []

    if style_results:
        style = style_results[0]
        keywords = style.get("Keywords", "")
        effects = style.get("Effects & Animation", "")

        if any(kw in keywords.lower() for kw in ["data", "dense", "dashboard", "grid"]):
            layout["Max Width"] = "1400px or full-width"
            layout["Grid"] = "12-column grid for data flexibility"
            spacing["Content Density"] = "High — optimize for information display"
        elif any(kw in keywords.lower() for kw in ["minimal", "simple", "clean", "single"]):
            layout["Max Width"] = "800px (narrow, focused)"
            layout["Layout"] = "Single column, centered"
            spacing["Content Density"] = "Low — focus on clarity"
        else:
            layout["Max Width"] = "1200px (standard)"
            layout["Layout"] = "Full-width sections, centered content"

        if effects:
            recommendations.append(f"Effects: {effects}")

    for ux in ux_results:
        category = ux.get("Category", "")
        do_text = ux.get("Do", "")
        dont_text = ux.get("Don't", "")
        if do_text:
            recommendations.append(f"{category}: {do_text}")
        if dont_text:
            components.append(f"Avoid: {dont_text}")

    if landing_results:
        landing = landing_results[0]
        sections = landing.get("Section Order", "")
        cta_placement = landing.get("Primary CTA Placement", "")
        color_strategy = landing.get("Color Strategy", "")

        if sections:
            layout["Sections"] = sections
        if cta_placement:
            recommendations.append(f"CTA Placement: {cta_placement}")
        if color_strategy:
            colors["Strategy"] = color_strategy

    if not layout:
        layout["Max Width"] = "1200px"
        layout["Layout"] = "Responsive grid"

    if not recommendations:
        recommendations = [
            "Refer to MASTER.md for all design rules",
            "Add specific overrides as needed for this page"
        ]

    return {
        "page_type": page_type,
        "layout": layout,
        "spacing": spacing,
        "typography": typography,
        "colors": colors,
        "components": components,
        "unique_components": unique_components,
        "recommendations": recommendations
    }


def _detect_page_type(context: str, style_results: list) -> str:
    context_lower = context.lower()

    page_patterns = [
        (["dashboard", "admin", "analytics", "data", "metrics", "stats", "monitor", "overview"], "Dashboard / Data View"),
        (["checkout", "payment", "cart", "purchase", "order", "billing"], "Checkout / Payment"),
        (["settings", "profile", "account", "preferences", "config"], "Settings / Profile"),
        (["landing", "marketing", "homepage", "hero", "home", "promo"], "Landing / Marketing"),
        (["login", "signin", "signup", "register", "auth", "password"], "Authentication"),
        (["pricing", "plans", "subscription", "tiers", "packages"], "Pricing / Plans"),
        (["blog", "article", "post", "news", "content", "story"], "Blog / Article"),
        (["product", "item", "detail", "pdp", "shop", "store"], "Product Detail"),
        (["search", "results", "browse", "filter", "catalog", "list"], "Search Results"),
        (["empty", "404", "error", "not found", "zero"], "Empty State"),
    ]

    for keywords, page_type in page_patterns:
        if any(kw in context_lower for kw in keywords):
            return page_type

    if style_results:
        best_for = style_results[0].get("Best For", "").lower()
        if "dashboard" in best_for or "data" in best_for:
            return "Dashboard / Data View"
        elif "landing" in best_for or "marketing" in best_for:
            return "Landing / Marketing"

    return "General"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Design System")
    parser.add_argument("query", help="Search query (e.g., 'SaaS dashboard')")
    parser.add_argument("--project-name", "-p", type=str, default=None, help="Project name")
    parser.add_argument("--format", "-f", choices=["ascii", "markdown"], default="ascii", help="Output format")

    args = parser.parse_args()

    result = generate_design_system(args.query, args.project_name, args.format)
    print(result["text"])
