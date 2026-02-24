# Map-Reflect Paper

## Building the PDF

This paper uses [Tectonic](https://tectonic-typesetting.github.io/) for PDF generation.

### Install Tectonic

```bash
# macOS
brew install tectonic

# Linux
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
```

### Generate PDF

```bash
cd paper
tectonic vldb.tex
```

This produces `vldb.pdf`.

## Regenerating Charts

The paper includes three visualizations generated from experimental results:

1. `chart_api_field_gap.png` - API vs Field accuracy dumbbell chart
2. `chart_method_sensitivity.png` - Method sensitivity to LLM choice
3. `chart_signal_heatmap.png` - Reflection signal performance heatmap

To regenerate:

```bash
cd paper
python generate_charts_final.py
```

Data sources for each chart are documented in comments at the top of each function in `generate_charts_final.py`.

## Files

- `vldb.tex` - Main paper source
- `sample.bib` - Bibliography
- `vldb.cls` - VLDB conference style
- `generate_charts_final.py` - Chart generation script
- `chart_integration_note.md` - Chart integration instructions
