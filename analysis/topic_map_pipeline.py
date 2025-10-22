"""
Topic map pipeline for seminar transcripts.

The script ingests all `.txt` files in an input directory, performs lightweight
topic modeling, and produces CSV summaries plus an interactive topic network
HTML. Re-run the script whenever new transcripts are added.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Transcript:
    """Container for raw transcript metadata and content."""

    name: str
    path: Path
    text: str


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #

def load_transcripts(input_dir: Path) -> List[Transcript]:
    """Load all `.txt` files from the input directory."""
    transcripts: List[Transcript] = []
    for path in sorted(input_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 for the odd file with accents.
            text = path.read_text(encoding="latin-1")
        transcripts.append(Transcript(name=path.stem, path=path, text=text))

    if not transcripts:
        raise ValueError(f"No .txt transcripts found in {input_dir}")

    return transcripts


def vectorize_transcripts(transcripts: Iterable[Transcript], max_features: int) -> Tuple[np.ndarray, List[str], TfidfVectorizer]:
    """Convert transcripts to a TF-IDF matrix."""
    texts = [item.text for item in transcripts]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.9,
        min_df=2,
        max_features=max_features,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    return matrix, feature_names, vectorizer


# --------------------------------------------------------------------------- #
# Topic modeling
# --------------------------------------------------------------------------- #

def compute_topics(tfidf_matrix: np.ndarray, num_topics: int, random_state: int) -> NMF:
    """Fit an NMF model on top of the TF-IDF features."""
    model = NMF(
        n_components=num_topics,
        init="nndsvda",
        random_state=random_state,
        beta_loss="frobenius",
        max_iter=400,
    )
    model.fit(tfidf_matrix)
    return model


def summarize_topics(model: NMF, feature_names: List[str], top_terms: int) -> pd.DataFrame:
    """Return a dataframe with the top terms per topic."""
    rows = []
    for topic_idx, component in enumerate(model.components_):
        top_indices = np.argsort(component)[::-1][:top_terms]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = component[top_indices]
        rows.append(
            {
                "topic_id": topic_idx,
                "top_terms": ", ".join(top_words),
                "top_terms_json": json.dumps(
                    [{"term": word, "weight": float(weight)} for word, weight in zip(top_words, top_weights)]
                ),
                "top_terms_list": top_words,
                "top_weights_list": top_weights.tolist(),
            }
        )
    return pd.DataFrame(rows)


def document_topic_matrix(model: NMF, tfidf_matrix: np.ndarray) -> np.ndarray:
    """Get the document-topic distribution."""
    return model.transform(tfidf_matrix)


# --------------------------------------------------------------------------- #
# Graph building
# --------------------------------------------------------------------------- #

def build_topic_graph(
    transcripts: List[Transcript],
    doc_topic: np.ndarray,
    topic_df: pd.DataFrame,
    top_topics_per_doc: int,
    min_edge_weight: float,
) -> nx.Graph:
    """
    Build a bipartite graph connecting transcript nodes to topic nodes.

    Edges keep the normalized topic weights used during visualization.
    """
    graph = nx.Graph()

    for transcript_idx, transcript in enumerate(transcripts):
        graph.add_node(
            f"doc::{transcript_idx}",
            label=transcript.name,
            type="transcript",
        )

    for _, row in topic_df.iterrows():
        graph.add_node(
            f"topic::{row.topic_id}",
            label=f"Topic {row.topic_id}",
            type="topic",
            terms=row.top_terms,
        )

    # Normalize weights per document to improve comparability.
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    normalized = np.divide(
        doc_topic,
        np.where(row_sums == 0, 1, row_sums),
        out=np.zeros_like(doc_topic),
        where=row_sums != 0,
    )

    for doc_idx in range(doc_topic.shape[0]):
        weights = normalized[doc_idx]
        if top_topics_per_doc:
            candidate_indices = np.argsort(weights)[::-1][:top_topics_per_doc]
        else:
            candidate_indices = np.where(weights >= min_edge_weight)[0]

        for topic_idx in candidate_indices:
            weight = weights[topic_idx]
            if weight < min_edge_weight:
                continue
            graph.add_edge(
                f"doc::{doc_idx}",
                f"topic::{topic_idx}",
                weight=float(weight),
            )

    return graph


def build_topic_network_figure(graph: nx.Graph) -> go.Figure:
    """Render the graph using Plotly."""
    layout = nx.spring_layout(graph, seed=42, k=None)

    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_weights: List[float] = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(data.get("weight", 0.0))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    transcript_x: List[float] = []
    transcript_y: List[float] = []
    transcript_text: List[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "transcript":
            continue
        x, y = layout[node]
        transcript_x.append(x)
        transcript_y.append(y)
        transcript_text.append(data.get("label", node))

    transcript_trace = go.Scatter(
        x=transcript_x,
        y=transcript_y,
        mode="markers",
        marker=dict(size=12, color="#1f77b4"),
        hoverinfo="text",
        name="Seminars",
        text=transcript_text,
    )

    topic_x: List[float] = []
    topic_y: List[float] = []
    topic_text: List[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "topic":
            continue
        x, y = layout[node]
        topic_x.append(x)
        topic_y.append(y)
        terms_preview = data.get("terms", "")
        topic_text.append(f"{data.get('label')}<br>{terms_preview}")

    topic_trace = go.Scatter(
        x=topic_x,
        y=topic_y,
        mode="markers",
        marker=dict(size=18, color="#ff7f0e", symbol="diamond"),
        hoverinfo="text",
        name="Topics",
        text=topic_text,
    )

    fig = go.Figure(
        data=[edge_trace, transcript_trace, topic_trace],
        layout=go.Layout(
            title="Metagov Seminar Topic Map",
            showlegend=True,
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def build_topic_heatmap_figure(doc_topic: np.ndarray, transcripts: List[Transcript]) -> Optional[go.Figure]:
    """Create a transcript-topic heatmap."""
    if doc_topic.size == 0:
        return None

    normalized = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-9)

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized,
            x=[f"Topic {idx}" for idx in range(doc_topic.shape[1])],
            y=[transcript.name for transcript in transcripts],
            colorscale="Blues",
        )
    )
    fig.update_layout(
        title="Transcript vs Topic Intensity",
        xaxis_title="Topic",
        yaxis_title="Transcript",
        height=max(600, 20 * doc_topic.shape[0]),
    )
    return fig


def build_transcript_scatter_figure(doc_topic: np.ndarray, transcripts: List[Transcript]) -> Optional[go.Figure]:
    """Create a 2D scatter plot of transcripts using PCA on topic weights."""
    if doc_topic.shape[0] < 2:
        return None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(doc_topic)
    explained = pca.explained_variance_ratio_

    fig = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=[transcript.name for transcript in transcripts],
            textposition="top center",
            marker=dict(size=10, color="#2ca02c"),
            hovertemplate="<b>%{text}</b><br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Transcript Similarity (PCA on Topic Weights)",
        xaxis_title=f"PCA 1 ({explained[0]*100:.1f}% var)",
        yaxis_title=f"PCA 2 ({explained[1]*100:.1f}% var)",
        showlegend=False,
    )
    return fig


def build_topic_term_dropdown_figure(topic_df: pd.DataFrame) -> Optional[go.Figure]:
    """Interactive bar chart to browse top terms per topic."""
    if topic_df.empty:
        return None

    terms_per_topic = []
    for _, row in topic_df.iterrows():
        try:
            entries = json.loads(row.top_terms_json)
        except json.JSONDecodeError:
            entries = []
        x = [entry["term"] for entry in entries]
        y = [entry["weight"] for entry in entries]
        terms_per_topic.append((row.topic_id, x, y))

    initial_topic, initial_x, initial_y = terms_per_topic[0]

    fig = go.Figure(
        data=go.Bar(x=initial_x, y=initial_y, marker_color="#d62728")
    )

    buttons = []
    for topic_id, x_values, y_values in terms_per_topic:
        buttons.append(
            dict(
                label=f"Topic {topic_id}",
                method="update",
                args=[
                    {"x": [x_values], "y": [y_values]},
                    {"title": f"Top Terms for Topic {topic_id}"},
                ],
            )
        )

    fig.update_layout(
        title=f"Top Terms for Topic {initial_topic}",
        xaxis_title="Term",
        yaxis_title="Weight",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
            )
        ],
    )
    return fig


def build_topic_sunburst_figure(
    doc_topic: np.ndarray,
    transcripts: List[Transcript],
    topic_df: pd.DataFrame,
) -> Optional[go.Figure]:
    """Create a sunburst highlighting dominant topic per transcript."""
    if doc_topic.size == 0 or topic_df.empty:
        return None

    normalized = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-9)
    root_label = "Metagov Seminars"

    labels: List[str] = [root_label]
    parents: List[str] = [""]
    values: List[float] = [float(normalized.sum())]
    hovertext: List[str] = [f"Total transcripts: {len(transcripts)}"]

    topic_terms = {
        int(row.topic_id): row.top_terms
        for _, row in topic_df.iterrows()
    }

    topic_totals = normalized.sum(axis=0)
    for topic_idx, total in enumerate(topic_totals):
        labels.append(f"Topic {topic_idx}")
        parents.append(root_label)
        values.append(float(total))
        hovertext.append(
            f"<b>Topic {topic_idx}</b><br>Total weight: {total:.2f}<br>Top terms: {topic_terms.get(topic_idx, '')}"
        )

    for doc_idx, transcript in enumerate(transcripts):
        if normalized.shape[1] == 0:
            continue
        topic_idx = int(np.argmax(normalized[doc_idx]))
        weight = float(normalized[doc_idx, topic_idx])
        if weight <= 0:
            continue
        labels.append(transcript.name)
        parents.append(f"Topic {topic_idx}")
        hovertext.append(
            f"<b>{transcript.name}</b><br>Assigned topic: {topic_idx}<br>Weight: {weight:.2f}"
        )
        values.append(weight)

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hoverinfo="text",
            hovertext=hovertext,
            maxdepth=3,
        )
    )
    fig.update_layout(
        title="Metagov Topic Galaxy",
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


def write_dashboard_html(figures: List[Tuple[str, str, go.Figure]], output_html: Path) -> None:
    """Combine multiple Plotly figures into a single HTML dashboard."""
    sections = []
    nav_links = []

    for title, slug, figure in figures:
        figure_html = pio.to_html(figure, include_plotlyjs=False, full_html=False)
        sections.append(
            f'<section id="{slug}" class="viz-section"><h2>{title}</h2>{figure_html}</section>'
        )
        nav_links.append(f'<a href="#{slug}">{title}</a>')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Metagov Seminar Insights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
    }}
    header {{
      padding: 24px 32px;
      background: linear-gradient(135deg, #1e3a8a, #312e81);
      box-shadow: 0 4px 20px rgba(15, 23, 42, 0.45);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 2.4rem;
    }}
    p.subtitle {{
      margin: 0;
      color: #cbd5f5;
      max-width: 900px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 16px 32px;
      background: rgba(15, 23, 42, 0.85);
      position: sticky;
      top: 0;
      backdrop-filter: blur(6px);
      z-index: 5;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.35);
    }}
    nav a {{
      color: #93c5fd;
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 1rem;
      border: 1px solid rgba(147, 197, 253, 0.2);
      transition: all 0.2s ease;
    }}
    nav a:hover {{
      background: rgba(59, 130, 246, 0.2);
      color: #bfdbfe;
    }}
    section.viz-section {{
      padding: 32px;
    }}
    section.viz-section h2 {{
      font-size: 1.8rem;
      margin-bottom: 18px;
      color: #bfdbfe;
      text-shadow: 0 2px 12px rgba(59, 130, 246, 0.4);
    }}
    .viz-section:nth-of-type(odd) {{
      background: rgba(30, 41, 59, 0.65);
    }}
    .viz-section:nth-of-type(even) {{
      background: rgba(15, 23, 42, 0.65);
    }}
    footer {{
      text-align: center;
      padding: 24px 16px 40px;
      color: #64748b;
      font-size: 0.9rem;
    }}
    @media (max-width: 768px) {{
      section.viz-section {{
        padding: 20px;
      }}
      h1 {{
        font-size: 2rem;
      }}
      nav {{
        justify-content: center;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Metagov Seminar Topic Observatory</h1>
    <p class="subtitle">
      Explore the thematic landscape of the Metagov seminar series. Zoom, hover, and jump between
      views to uncover how conversations intertwine across time and topics.
    </p>
  </header>
  <nav>
    {' '.join(nav_links)}
  </nav>
  <main>
    {''.join(sections)}
  </main>
  <footer>
    Generated with the Metagov topic map pipeline. Re-run after adding new transcripts to refresh insights.
  </footer>
</body>
</html>
"""

    output_html.write_text(html_content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    num_topics: int,
    top_terms: int,
    top_topics_per_doc: int,
    min_edge_weight: float,
    max_features: int,
    random_state: int,
) -> None:
    """End-to-end pipeline."""
    transcripts = load_transcripts(input_dir)

    tfidf_matrix, feature_names, _ = vectorize_transcripts(transcripts, max_features=max_features)
    model = compute_topics(tfidf_matrix, num_topics=num_topics, random_state=random_state)

    topic_df = summarize_topics(model, feature_names, top_terms=top_terms)
    doc_topic = document_topic_matrix(model, tfidf_matrix)

    docs_df = pd.DataFrame(
        {
            "transcript": [transcript.name for transcript in transcripts],
            **{f"topic_{idx}": doc_topic[:, idx] for idx in range(doc_topic.shape[1])},
        }
    )

    graph = build_topic_graph(
        transcripts=transcripts,
        doc_topic=doc_topic,
        topic_df=topic_df,
        top_topics_per_doc=top_topics_per_doc,
        min_edge_weight=min_edge_weight,
    )

    network_fig = build_topic_network_figure(graph)
    heatmap_fig = build_topic_heatmap_figure(doc_topic, transcripts)
    scatter_fig = build_transcript_scatter_figure(doc_topic, transcripts)
    term_browser_fig = build_topic_term_dropdown_figure(topic_df)
    sunburst_fig = build_topic_sunburst_figure(doc_topic, transcripts, topic_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    topic_df.to_csv(data_dir / "topic_terms.csv", index=False)
    docs_df.to_csv(data_dir / "document_topic_weights.csv", index=False)
    figures = [
        ("Topic Network", "topic-network", "topic_network.html", network_fig),
        ("Transcript vs Topic Heatmap", "topic-heatmap", "topic_heatmap.html", heatmap_fig),
        ("Transcript Similarity Map", "transcript-scatter", "transcript_scatter.html", scatter_fig),
        ("Topic Term Explorer", "topic-browser", "topic_term_browser.html", term_browser_fig),
        ("Topic Galaxy Sunburst", "topic-sunburst", "topic_sunburst.html", sunburst_fig),
    ]

    for title, slug, filename, figure in figures:
        if figure is None:
            continue
        figure.write_html(viz_dir / filename, include_plotlyjs="cdn")

    available_figures = [(title, slug, fig) for title, slug, _, fig in figures if fig is not None]
    if available_figures:
        write_dashboard_html(available_figures, viz_dir / "topic_dashboard.html")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a topic map for seminar transcripts.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing .txt transcripts.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"), help="Where processed assets are stored.")
    parser.add_argument("--num-topics", type=int, default=12, help="Number of topics to extract with NMF.")
    parser.add_argument("--top-terms", type=int, default=12, help="Number of terms to keep in topic summaries.")
    parser.add_argument("--top-topics-per-doc", type=int, default=3, help="Edges per transcript (set 0 to use min-edge-weight).")
    parser.add_argument("--min-edge-weight", type=float, default=0.05, help="Minimum normalized weight for doc-topic edges.")
    parser.add_argument("--max-features", type=int, default=8000, help="Maximum vocabulary size for TF-IDF.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used in the modeling steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        num_topics=args.num_topics,
        top_terms=args.top_terms,
        top_topics_per_doc=args.top_topics_per_doc,
        min_edge_weight=args.min_edge_weight,
        max_features=args.max_features,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
