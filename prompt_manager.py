"""
Prompt Manager — A modern desktop app to save, organize, and retrieve AI prompts.
Built with CustomTkinter, JSON storage, and pyperclip.

Architecture:
  - DataManager  : handles all JSON read/write operations (Data Layer)
  - PromptManagerApp : builds and manages the UI (Presentation Layer)
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Optional

import customtkinter as ctk
import pyperclip
import google.generativeai as genai

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

APP_TITLE       = "Prompt Manager"
APP_VERSION     = "1.0.0"
DATA_FILE       = "prompts.json"
WINDOW_MIN_W    = 900
WINDOW_MIN_H    = 600
SIDEBAR_W       = 280
ACCENT          = "#7B6FF0"        # violet accent
ACCENT_HOVER    = "#9D94F5"
SUCCESS         = "#3FCF8E"        # green for "Copied"
DANGER          = "#F05F5F"        # red for delete confirm
BG_DARK         = "#0F0F14"
BG_PANEL        = "#16161E"
BG_SIDEBAR      = "#111118"
BG_ENTRY        = "#1E1E2A"
TEXT_PRIMARY    = "#E8E8F0"
TEXT_SECONDARY  = "#7A7A9A"
PLACEHOLDER_RE  = re.compile(r'\[([^\]]+)\]')   # matches [Variable Name]

# ─────────────────────────────────────────────
#  GEMINI AI CONFIGURATION
# ─────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────
#  AI HELPER FUNCTIONS
# ─────────────────────────────────────────────

def generate_title_and_tags(body: str) -> tuple[str, str]:
    """Use Gemini AI to generate a title and tags from the prompt body."""
    if not GEMINI_API_KEY or not body.strip():
        return "", ""
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""Based on the following prompt/instruction, generate:
1. A concise title (max 50 characters)
2. Comma-separated tags (3-5 relevant tags)

Prompt:
{body}

Respond in this exact format:
Title: [your title here]
Tags: [tag1, tag2, tag3]"""
        
        response = model.generate_content(prompt, stream=False)
        text = response.text.strip()
        
        # Parse response
        lines = text.split("\n")
        title = ""
        tags = ""
        
        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Tags:"):
                tags = line.replace("Tags:", "").strip()
        
        return title, tags
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "", ""


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

class DataManager:
    """Handles all persistence operations against prompts.json."""

    def __init__(self, filepath: str = DATA_FILE) -> None:
        self.filepath = filepath
        self._data: dict[str, dict] = {}
        self._load()

    # ── internal ──────────────────────────────

    def _load(self) -> None:
        """Load prompts from disk, creating the file if absent."""
        if not os.path.exists(self.filepath):
            self._data = {}
            self._persist()
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # Accept both list-of-dicts (legacy) and dict-of-dicts (current)
            if isinstance(raw, list):
                self._data = {item["id"]: item for item in raw}
            elif isinstance(raw, dict):
                self._data = raw
            else:
                self._data = {}
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted file — back it up and start fresh
            backup = self.filepath + ".bak"
            if os.path.exists(self.filepath):
                os.rename(self.filepath, backup)
            self._data = {}
            self._persist()

    def _persist(self) -> None:
        """Write current state to disk (thread-safe via a background thread)."""
        data_snapshot = dict(self._data)

        def _write():
            try:
                with open(self.filepath, "w", encoding="utf-8") as fh:
                    json.dump(data_snapshot, fh, indent=2, ensure_ascii=False)
            except OSError:
                pass  # Silently fail on write errors; data stays in memory

        threading.Thread(target=_write, daemon=True).start()

    # ── public API ────────────────────────────

    def all_prompts(self) -> list[dict]:
        """Return all prompts sorted by title (case-insensitive)."""
        return sorted(self._data.values(), key=lambda p: p.get("title", "").lower())

    def get(self, prompt_id: str) -> Optional[dict]:
        """Fetch a single prompt by id."""
        return self._data.get(prompt_id)

    def create(self, title: str, body: str, tags: str = "") -> dict:
        """Create a new prompt entry and persist it."""
        prompt = {
            "id":    str(uuid.uuid4()),
            "title": title.strip() or "Untitled",
            "body":  body,
            "tags":  tags.strip(),
        }
        self._data[prompt["id"]] = prompt
        self._persist()
        return prompt

    def update(self, prompt_id: str, title: str, body: str, tags: str = "") -> bool:
        """Update an existing prompt. Returns False if id not found."""
        if prompt_id not in self._data:
            return False
        self._data[prompt_id].update({
            "title": title.strip() or "Untitled",
            "body":  body,
            "tags":  tags.strip(),
        })
        self._persist()
        return True

    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt by id. Returns False if id not found."""
        if prompt_id not in self._data:
            return False
        del self._data[prompt_id]
        self._persist()
        return True

    def search(self, query: str) -> list[dict]:
        """Filter prompts whose title or tags contain the query string."""
        q = query.lower().strip()
        if not q:
            return self.all_prompts()
        return [
            p for p in self.all_prompts()
            if q in p.get("title", "").lower() or q in p.get("tags", "").lower()
        ]


# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────

def _make_font(size: int = 13, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


def _accent_button(parent, text: str, command, width: int = 120, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=ACCENT, hover_color=ACCENT_HOVER,
        text_color=TEXT_PRIMARY, corner_radius=8,
        font=_make_font(13, "bold"), width=width, **kw,
    )


def _ghost_button(parent, text: str, command, width: int = 100, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color="transparent", hover_color=BG_ENTRY,
        text_color=TEXT_SECONDARY, corner_radius=8,
        border_width=1, border_color=BG_ENTRY,
        font=_make_font(12), width=width, **kw,
    )


# ─────────────────────────────────────────────
#  DIALOG: New / Edit Prompt
# ─────────────────────────────────────────────

class PromptDialog(ctk.CTkToplevel):
    """Modal dialog for creating or editing a prompt."""

    def __init__(
        self,
        parent,
        on_save,
        title: str = "",
        body: str = "",
        tags: str = "",
        prompt_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.on_save   = on_save
        self.prompt_id = prompt_id

        mode = "Edit Prompt" if prompt_id else "New Prompt"
        self.title(mode)
        self.geometry("680x540")
        self.resizable(True, True)
        self.configure(fg_color=BG_PANEL)
        self.grab_set()   # modal
        self.focus()

        self._build(title, body, tags, mode)

    def _build(self, title: str, body: str, tags: str, mode: str) -> None:
        pad = {"padx": 24, "pady": 6}

        # Header
        ctk.CTkLabel(
            self, text=mode, font=_make_font(18, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(20, 4))

        ctk.CTkLabel(
            self, text="Title", font=_make_font(12, "bold"),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", **pad)
        self.title_entry = ctk.CTkEntry(
            self, placeholder_text="e.g. Code Review Assistant",
            fg_color=BG_ENTRY, border_color=BG_ENTRY,
            text_color=TEXT_PRIMARY, font=_make_font(13),
            height=38, corner_radius=8,
        )
        self.title_entry.pack(fill="x", padx=24, pady=(0, 8))
        self.title_entry.insert(0, title)

        ctk.CTkLabel(
            self, text="Tags  (comma-separated, optional)", font=_make_font(12, "bold"),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", **pad)
        self.tags_entry = ctk.CTkEntry(
            self, placeholder_text="e.g. coding, review, python",
            fg_color=BG_ENTRY, border_color=BG_ENTRY,
            text_color=TEXT_PRIMARY, font=_make_font(12),
            height=34, corner_radius=8,
        )
        self.tags_entry.pack(fill="x", padx=24, pady=(0, 8))
        self.tags_entry.insert(0, tags)

        hint = (
            "💡 Use [Variable Name] placeholders — e.g. [Insert Code Here], [Language]"
        )
        ctk.CTkLabel(
            self, text=hint, font=_make_font(11),
            text_color=TEXT_SECONDARY, wraplength=620, justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 4))

        ctk.CTkLabel(
            self, text="Prompt Body", font=_make_font(12, "bold"),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", **pad)
        
        # Body text with AI generation button
        body_frame = ctk.CTkFrame(self, fg_color="transparent")
        body_frame.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        body_frame.grid_columnconfigure(0, weight=1)
        
        self.body_text = ctk.CTkTextbox(
            body_frame, fg_color=BG_ENTRY, border_color=BG_ENTRY,
            text_color=TEXT_PRIMARY, font=_make_font(13),
            corner_radius=8, wrap="word",
        )
        self.body_text.grid(row=0, column=0, sticky="nsew")
        body_frame.grid_rowconfigure(0, weight=1)
        self.body_text.insert("1.0", body)
        
        # AI Generation button
        if GEMINI_API_KEY:
            ai_btn = ctk.CTkButton(
                body_frame, text="✨ Generate Title & Tags",
                command=self._generate_with_ai,
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                text_color=TEXT_PRIMARY, corner_radius=8,
                font=_make_font(11, "bold"), height=32,
            )
            ai_btn.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=29, pady=(0, 12))
        _ghost_button(btn_row, "Cancel", self.destroy, width=90).pack(side="right", padx=(8, 0))
        _accent_button(btn_row, "Save Prompt", self._save, width=140).pack(side="right")

    def _save(self) -> None:
        title = self.title_entry.get().strip()
        body  = self.body_text.get("1.0", "end-1c").strip()
        tags  = self.tags_entry.get().strip()
        if not title:
            self.title_entry.configure(border_color=DANGER)
            return
        self.on_save(title=title, body=body, tags=tags, prompt_id=self.prompt_id)
        self.destroy()

    def _generate_with_ai(self) -> None:
        """Generate title and tags using Gemini AI in a background thread."""
        body = self.body_text.get("1.0", "end-1c").strip()
        if not body:
            return
        
        # Show loading state
        self.title_entry.configure(state="disabled")
        self.tags_entry.configure(state="disabled")
        
        def _run_ai():
            title, tags = generate_title_and_tags(body)
            # Update UI in main thread
            self.after(0, lambda: self._update_with_ai_result(title, tags))
        
        thread = threading.Thread(target=_run_ai, daemon=True)
        thread.start()

    def _update_with_ai_result(self, title: str, tags: str) -> None:
        """Update title and tags fields with AI-generated content."""
        # Re-enable fields
        self.title_entry.configure(state="normal")
        self.tags_entry.configure(state="normal")
        
        # Clear and populate fields
        self.title_entry.delete(0, "end")
        if title:
            self.title_entry.insert(0, title)
        
        self.tags_entry.delete(0, "end")
        if tags:
            self.tags_entry.insert(0, tags)


# ─────────────────────────────────────────────
#  DIALOG: Fill Variables
# ─────────────────────────────────────────────

class FillVariablesDialog(ctk.CTkToplevel):
    """Let the user fill in [placeholder] values before copying."""

    def __init__(self, parent, body: str, on_copy) -> None:
        super().__init__(parent)
        self.title("Fill in Variables")
        self.configure(fg_color=BG_PANEL)
        self.grab_set()
        self.focus()

        self._body    = body
        self._on_copy = on_copy
        self._vars: dict[str, ctk.CTkEntry] = {}

        placeholders = list(dict.fromkeys(PLACEHOLDER_RE.findall(body)))  # unique, ordered
        h = min(120 + len(placeholders) * 70, 600)
        self.geometry(f"520x{h}")
        self._build(placeholders)

    def _build(self, placeholders: list[str]) -> None:
        ctk.CTkLabel(
            self, text="Fill in Placeholders",
            font=_make_font(16, "bold"), text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(20, 4))

        ctk.CTkLabel(
            self,
            text="These values will replace [brackets] in your prompt before copying.",
            font=_make_font(11), text_color=TEXT_SECONDARY, wraplength=470, justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 12))

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24)

        for ph in placeholders:
            ctk.CTkLabel(
                scroll, text=f"[{ph}]", font=_make_font(12, "bold"),
                text_color=ACCENT,
            ).pack(anchor="w", pady=(8, 2))
            entry = ctk.CTkEntry(
                scroll, placeholder_text=f"Enter value for {ph}…",
                fg_color=BG_ENTRY, border_color=BG_ENTRY,
                text_color=TEXT_PRIMARY, font=_make_font(13),
                height=36, corner_radius=8,
            )
            entry.pack(fill="x")
            self._vars[ph] = entry

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=16)
        _ghost_button(btn_row, "Cancel", self.destroy, width=80).pack(side="right", padx=(8, 0))
        _accent_button(btn_row, "Copy Filled Prompt", self._copy, width=180).pack(side="right")

    def _copy(self) -> None:
        result = self._body
        for ph, entry in self._vars.items():
            val = entry.get() or f"[{ph}]"
            result = result.replace(f"[{ph}]", val)
        self._on_copy(result)
        self.destroy()


# ─────────────────────────────────────────────
#  SIDEBAR ITEM
# ─────────────────────────────────────────────

class SidebarItem(ctk.CTkFrame):
    """A single clickable row in the sidebar list."""

    def __init__(self, parent, prompt: dict, on_select, **kw) -> None:
        super().__init__(
            parent, fg_color="transparent",
            cursor="hand2", corner_radius=8, **kw,
        )
        self.prompt    = prompt
        self.on_select = on_select
        self._selected = False
        self._build()

    def _build(self) -> None:
        self.configure(height=52)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=8, pady=4)

        title_lbl = ctk.CTkLabel(
            inner, text=self.prompt.get("title", "Untitled"),
            font=_make_font(13, "bold"), text_color=TEXT_PRIMARY,
            anchor="w", wraplength=200,
        )
        title_lbl.pack(fill="x")

        tags = self.prompt.get("tags", "")
        if tags:
            ctk.CTkLabel(
                inner, text=tags, font=_make_font(10),
                text_color=TEXT_SECONDARY, anchor="w",
            ).pack(fill="x")

        # bind clicks on all children
        for w in (self, inner, title_lbl):
            w.bind("<Button-1>", self._clicked)

    def _clicked(self, _event=None) -> None:
        self.on_select(self.prompt["id"])

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        color = ACCENT if selected else "transparent"
        hover = ACCENT_HOVER if selected else BG_ENTRY
        self.configure(fg_color=color)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class PromptManagerApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.geometry("1200x720")
        self.minsize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.configure(fg_color=BG_DARK)

        self._dm              = DataManager()
        self._current_id: Optional[str] = None
        self._sidebar_items: list[SidebarItem] = []
        self._copy_timer: Optional[str] = None

        self._build_ui()
        self._refresh_sidebar()
        self._show_welcome()

    # ── UI CONSTRUCTION ───────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=SIDEBAR_W, fg_color=BG_SIDEBAR, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # ── Logo bar ──
        logo_bar = ctk.CTkFrame(sidebar, fg_color="transparent", height=56)
        logo_bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        logo_bar.grid_propagate(False)

        ctk.CTkLabel(
            logo_bar, text="⚡ Prompt Manager",
            font=_make_font(15, "bold"), text_color=TEXT_PRIMARY,
        ).pack(side="left", padx=4, pady=12)

        # ── Search ──
        search_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        search_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(8, 4))

        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", self._on_search)

        search_entry = ctk.CTkEntry(
            search_frame, textvariable=self._search_var,
            placeholder_text="🔍  Search prompts…",
            fg_color=BG_ENTRY, border_color=BG_ENTRY,
            text_color=TEXT_PRIMARY, font=_make_font(12),
            height=36, corner_radius=8,
        )
        search_entry.pack(fill="x")

        # ── Prompt list ──
        self._list_frame = ctk.CTkScrollableFrame(
            sidebar, fg_color="transparent",
            scrollbar_button_color=BG_ENTRY,
            scrollbar_button_hover_color=ACCENT,
        )
        self._list_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)

        # ── New prompt button ──
        new_btn = _accent_button(
            sidebar, "＋  New Prompt", self._open_new_dialog,
            width=SIDEBAR_W - 24, height=40,
        )
        new_btn.grid(row=3, column=0, padx=12, pady=(4, 14))

    def _build_main_panel(self) -> None:
        main = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # ── Top action bar ──
        self._action_bar = ctk.CTkFrame(main, fg_color="transparent", height=60)
        self._action_bar.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 0))
        self._action_bar.grid_propagate(False)
        self._action_bar.grid_columnconfigure(0, weight=1)

        self._prompt_title_lbl = ctk.CTkLabel(
            self._action_bar, text="",
            font=_make_font(20, "bold"), text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self._prompt_title_lbl.grid(row=0, column=0, sticky="ew")

        btn_group = ctk.CTkFrame(self._action_bar, fg_color="transparent")
        btn_group.grid(row=0, column=1, sticky="e")

        self._copy_btn = _accent_button(
            btn_group, "📋  Copy", self._copy_prompt,
            width=120, height=36,
        )
        self._copy_btn.pack(side="right", padx=(8, 0))

        self._edit_btn = _ghost_button(
            btn_group, "✏️  Edit", self._open_edit_dialog,
            width=90, height=36,
        )
        self._edit_btn.pack(side="right", padx=(8, 0))

        self._delete_btn = ctk.CTkButton(
            btn_group, text="🗑  Delete", command=self._confirm_delete,
            fg_color="transparent", hover_color="#3A1A1A",
            text_color=DANGER, corner_radius=8,
            border_width=1, border_color=DANGER,
            font=_make_font(12), width=90, height=36,
        )
        self._delete_btn.pack(side="right", padx=(8, 0))

        # ── Tags label ──
        self._tags_lbl = ctk.CTkLabel(
            main, text="", font=_make_font(11),
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self._tags_lbl.grid(row=1, column=0, sticky="nw", padx=26, pady=(4, 0))

        # ── Body viewer ──
        self._body_box = ctk.CTkTextbox(
            main, fg_color=BG_ENTRY, border_color=BG_ENTRY,
            text_color=TEXT_PRIMARY, font=_make_font(14),
            corner_radius=12, wrap="word", state="disabled",
        )
        self._body_box.grid(row=2, column=0, sticky="nsew", padx=24, pady=(8, 0))
        main.grid_rowconfigure(2, weight=1)

        # ── Variables banner (hidden until prompt has placeholders) ──
        self._var_banner = ctk.CTkFrame(main, fg_color=BG_ENTRY, corner_radius=8)
        self._var_banner.grid(row=3, column=0, sticky="ew", padx=24, pady=12)
        self._var_banner.grid_columnconfigure(0, weight=1)
        self._var_banner.grid_remove()  # hidden by default

        self._var_label = ctk.CTkLabel(
            self._var_banner, text="",
            font=_make_font(12), text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self._var_label.grid(row=0, column=0, sticky="ew", padx=12, pady=8)

        _accent_button(
            self._var_banner, "Fill Variables & Copy",
            self._open_fill_dialog, width=190, height=34,
        ).grid(row=0, column=1, padx=12, pady=8)

    # ── SIDEBAR MANAGEMENT ────────────────────

    def _refresh_sidebar(self, query: str = "") -> None:
        """Re-render sidebar items based on optional search query."""
        for item in self._sidebar_items:
            item.destroy()
        self._sidebar_items.clear()

        prompts = self._dm.search(query)
        for p in prompts:
            item = SidebarItem(self._list_frame, p, self._load_prompt)
            item.pack(fill="x", pady=2)
            if p["id"] == self._current_id:
                item.set_selected(True)
            self._sidebar_items.append(item)

    def _on_search(self, *_args) -> None:
        self._refresh_sidebar(self._search_var.get())

    def _select_sidebar_item(self, prompt_id: str) -> None:
        for item in self._sidebar_items:
            item.set_selected(item.prompt["id"] == prompt_id)

    # ── PROMPT DISPLAY ────────────────────────

    def _show_welcome(self) -> None:
        self._prompt_title_lbl.configure(text="Welcome to Prompt Manager")
        self._tags_lbl.configure(text="Create a new prompt or select one from the sidebar.")
        self._body_box.configure(state="normal")
        self._body_box.delete("1.0", "end")
        welcome = (
            "👈  Select a prompt from the sidebar to view its content here.\n\n"
            "➕  Click 'New Prompt' to create your first prompt.\n\n"
            "📋  Use 'Copy' to instantly copy a prompt to your clipboard.\n\n"
            "🔍  Use the search bar to quickly find any prompt by title or tag."
        )
        self._body_box.insert("1.0", welcome)
        self._body_box.configure(state="disabled")
        self._copy_btn.configure(state="disabled")
        self._edit_btn.configure(state="disabled")
        self._delete_btn.configure(state="disabled")
        self._var_banner.grid_remove()
        self._current_id = None

    def _load_prompt(self, prompt_id: str) -> None:
        """Display a prompt in the main panel."""
        prompt = self._dm.get(prompt_id)
        if not prompt:
            return

        self._current_id = prompt_id
        self._select_sidebar_item(prompt_id)

        self._prompt_title_lbl.configure(text=prompt.get("title", ""))
        tags = prompt.get("tags", "")
        self._tags_lbl.configure(
            text=f"🏷  {tags}" if tags else ""
        )

        body = prompt.get("body", "")
        self._body_box.configure(state="normal")
        self._body_box.delete("1.0", "end")
        self._body_box.insert("1.0", body)
        self._body_box.configure(state="disabled")

        self._copy_btn.configure(state="normal")
        self._edit_btn.configure(state="normal")
        self._delete_btn.configure(state="normal")

        # Show variable banner if prompt has placeholders
        placeholders = PLACEHOLDER_RE.findall(body)
        if placeholders:
            unique = list(dict.fromkeys(placeholders))
            label = "📝 Variables detected:  " + "  •  ".join(f"[{p}]" for p in unique)
            self._var_label.configure(text=label)
            self._var_banner.grid()
        else:
            self._var_banner.grid_remove()

    # ── CRUD OPERATIONS ───────────────────────

    def _open_new_dialog(self) -> None:
        def on_save(title, body, tags, prompt_id=None):
            prompt = self._dm.create(title, body, tags)
            self._refresh_sidebar(self._search_var.get())
            self._load_prompt(prompt["id"])

        PromptDialog(self, on_save=on_save)

    def _open_edit_dialog(self) -> None:
        if not self._current_id:
            return
        prompt = self._dm.get(self._current_id)
        if not prompt:
            return

        def on_save(title, body, tags, prompt_id):
            self._dm.update(prompt_id, title, body, tags)
            self._refresh_sidebar(self._search_var.get())
            self._load_prompt(prompt_id)

        PromptDialog(
            self, on_save=on_save,
            title=prompt.get("title", ""),
            body=prompt.get("body", ""),
            tags=prompt.get("tags", ""),
            prompt_id=self._current_id,
        )

    def _confirm_delete(self) -> None:
        if not self._current_id:
            return
        prompt = self._dm.get(self._current_id)
        if not prompt:
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("Confirm Delete")
        dlg.geometry("380x160")
        dlg.configure(fg_color=BG_PANEL)
        dlg.grab_set()
        dlg.focus()

        ctk.CTkLabel(
            dlg, text=f"Delete \"{prompt.get('title', '')}\"?",
            font=_make_font(14, "bold"), text_color=TEXT_PRIMARY,
        ).pack(pady=(24, 4))
        ctk.CTkLabel(
            dlg, text="This action cannot be undone.",
            font=_make_font(12), text_color=TEXT_SECONDARY,
        ).pack()

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(pady=20)
        _ghost_button(row, "Cancel", dlg.destroy, width=80).pack(side="left", padx=8)
        ctk.CTkButton(
            row, text="Delete", command=lambda: self._do_delete(dlg),
            fg_color=DANGER, hover_color="#C04040",
            text_color="white", corner_radius=8,
            font=_make_font(13, "bold"), width=90,
        ).pack(side="left", padx=8)

    def _do_delete(self, dlg: ctk.CTkToplevel) -> None:
        dlg.destroy()
        if self._current_id:
            self._dm.delete(self._current_id)
            self._current_id = None
            self._refresh_sidebar(self._search_var.get())
            self._show_welcome()

    # ── COPY / CLIPBOARD ─────────────────────

    def _copy_prompt(self) -> None:
        if not self._current_id:
            return
        prompt = self._dm.get(self._current_id)
        if not prompt:
            return
        body = prompt.get("body", "")
        if PLACEHOLDER_RE.search(body):
            # Has variables — let user choose: plain copy or fill
            self._do_copy(body)   # copy as-is with brackets intact
        else:
            self._do_copy(body)

    def _open_fill_dialog(self) -> None:
        if not self._current_id:
            return
        prompt = self._dm.get(self._current_id)
        if not prompt:
            return
        FillVariablesDialog(self, body=prompt.get("body", ""), on_copy=self._do_copy)

    def _do_copy(self, text: str) -> None:
        """Copy text to clipboard and give visual feedback."""
        try:
            pyperclip.copy(text)
        except Exception:
            pass  # clipboard not available in some headless environments

        # Visual feedback
        if self._copy_timer:
            self.after_cancel(self._copy_timer)
        self._copy_btn.configure(
            text="✓  Copied!", fg_color=SUCCESS, hover_color=SUCCESS,
        )
        self._copy_timer = self.after(2000, self._reset_copy_btn)

    def _reset_copy_btn(self) -> None:
        self._copy_btn.configure(
            text="📋  Copy", fg_color=ACCENT, hover_color=ACCENT_HOVER,
        )
        self._copy_timer = None


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = PromptManagerApp()
    app.mainloop()