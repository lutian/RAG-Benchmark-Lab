"""Visual identity shared by every page (palette and typography of the reference design)."""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --rl-accent: #2563EB;
  --rl-accent-soft: #EFF4FF;
  --rl-text: #1F2937;
  --rl-muted: #6B7280;
  --rl-border: #E5E7EB;
  --rl-panel: #F6F7F9;
  --rl-badge-bg: #E7F6EC;
  --rl-badge-text: #1E7B43;
  --rl-highlight: #FDF3B0;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.stApp { color: var(--rl-text); }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

section[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid var(--rl-border);
}

.rl-card {
  border: 1px solid var(--rl-border);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 12px;
  background: #FFFFFF;
}
.rl-card.selected { background: var(--rl-accent-soft); border-color: #C7D7FE; }
.rl-card-title { font-weight: 600; font-size: 1rem; }
.rl-card-meta { color: var(--rl-muted); font-size: 0.8rem; margin-top: 2px; }

.rl-badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 500;
  background: var(--rl-badge-bg);
  color: var(--rl-badge-text);
  margin-left: 6px;
}
.rl-badge.neutral { background: var(--rl-panel); color: var(--rl-muted); }

.rl-term { color: var(--rl-accent); }
mark.rl-evidence { background: var(--rl-highlight); padding: 0 2px; border-radius: 3px; }
mark.rl-chunk-a { background: #DBEAFE; }
mark.rl-chunk-b { background: #FDF3B0; }
mark.rl-overlap { background: #C7F0D2; }
mark[class^="rl-"] { padding: 0; border-radius: 2px; color: inherit; }

.rl-preview {
  border: 1px solid var(--rl-border);
  border-radius: 12px;
  padding: 18px 20px;
  max-height: 70vh;
  overflow-y: auto;
  font-size: 0.86rem;
  line-height: 1.6;
  white-space: pre-wrap;
  background: #FFFFFF;
}

.rl-snippet { font-size: 0.85rem; line-height: 1.55; margin-top: 8px; white-space: pre-wrap; }

.rl-brand { font-weight: 700; font-size: 1.4rem; letter-spacing: -0.02em; }
.rl-brand span { color: var(--rl-accent); }
</style>
"""


def apply_theme() -> None:
    """Inject the shared CSS. Call once at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def brand() -> None:
    """Render the product wordmark."""
    st.markdown('<div class="rl-brand">rag<span>.</span>lab</div>', unsafe_allow_html=True)
