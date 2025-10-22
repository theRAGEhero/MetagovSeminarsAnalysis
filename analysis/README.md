# Topic Map Pipeline

This folder contains a lightweight Python workflow that converts the Metagov seminar transcripts into an interactive topic map.

## 1. Environment

Create (or activate) a Python 3.9+ environment and install the required packages:

```bash
pip install -r analysis/requirements.txt
```

If you prefer conda or uv, translate the requirements accordingly.

## 2. Generate the topic map

Run the pipeline from the transcripts directory (or pass paths explicitly):

```bash
python analysis/topic_map_pipeline.py --input-dir . --output-dir analysis_output
```

Outputs:

- `analysis_output/data/topic_terms.csv`: top words for each topic.
- `analysis_output/data/document_topic_weights.csv`: topic weights per seminar.
- `analysis_output/visualizations/topic_network.html`: interactive network linking seminars and topics.
- `analysis_output/visualizations/topic_heatmap.html`: heatmap of transcript-topic intensities.
- `analysis_output/visualizations/transcript_scatter.html`: PCA scatter locating seminars by topical similarity.
- `analysis_output/visualizations/topic_term_browser.html`: dropdown bar chart to browse top terms per topic.
- `analysis_output/visualizations/topic_sunburst.html`: “Topic Galaxy” sunburst showing transcript distribution across dominant topics.
- `analysis_output/visualizations/topic_dashboard.html`: single-page dashboard that stitches all views together with navigation.

Open the HTML file in a browser to explore the map. Hover over nodes to inspect seminar names and dominant topic terms.

### Optional arguments

- `--num-topics`: change the number of latent topics (default: 12).
- `--top-topics-per-doc`: number of topic edges per seminar (default: 3).
- `--min-edge-weight`: minimum normalized weight for edges when `top-topics-per-doc` is set to 0.

## 3. Refreshing after adding transcripts

Drop new `.txt` files into the transcripts directory and re-run the same command. The script automatically ingests every transcript in the folder, retrains the topic model, and regenerates the outputs so you can keep the visualization current.

If you need higher fidelity clusters, experiment with alternative settings, or swap in a different model (e.g., `BERTopic`), the script is modular enough to adapt: replace the NMF block with your preferred model while keeping the input/output structure the same.
