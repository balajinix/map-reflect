# Map-Reflect Paper: Chart Integration Note
# For Claude Code

## Files

    generate_charts_final.py            the source script (run to regenerate)
    chart_api_field_gap.{pdf,png}       Figure A: dumbbell chart
    chart_method_sensitivity.{pdf,png}  Figure B: slope chart  (was: model_sensitivity)
    chart_signal_heatmap.{pdf,png}      Figure C: signal heatmap

All three images contain NO titles or captions. LaTeX adds those.
All figures are sized for VLDB single-column width (3.33 inches).


## When to regenerate

Run: python generate_charts_final.py

Each function has a "DATA" comment block at the top listing which paper
tables drive its numbers:

  chart_api_field_gap       Table 1 (API acc.) + Table 2 (Field F1), Llama-3.3 column
  chart_method_sensitivity  Table 2 (Field F1), both Granite-4 and Llama-3.3 columns
  chart_signal_heatmap      Table 3 (ITOpsGraph ablation) + Table 2 OMOP oracle rows


## LaTeX integration

### Step 1 — Copy PDFs to paper directory

    cp chart_api_field_gap.pdf     <paper_dir>/
    cp chart_method_sensitivity.pdf <paper_dir>/
    cp chart_signal_heatmap.pdf    <paper_dir>/

Use .pdf for camera-ready; .png is also available if the venue requires raster.


### Step 2 — Figure A: API-vs-Field gap dumbbell

Add as a new figure in Section 4 (Results), just before the first result
table (\begin{table*} for tab:api_accuracy_oracle). Insert:

    \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{chart_api_field_gap.pdf}
        \caption{%
            API-level accuracy (filled dot) vs.\ field-level F1 (hollow dot)
            per method and dataset in oracle mode (Llama-3.3-70B).
            The span of each segment is the \emph{last-mile gap} between
            coarse type selection and fine field alignment.
            Map-Reflect achieves consistently short spans; GNN and ReAct
            reach high API accuracy but near-zero field F1 on ITOpsGraph,
            indicating successful type identification without
            field-level understanding.
        }
        \label{fig:api_field_gap}
    \end{figure}

Then extend the sentence that begins "We evaluate schema mapping using
two complementary metrics" to read:

    "We evaluate schema mapping using two complementary metrics
     (Figure~\ref{fig:api_field_gap} visualises the gap between them
     across all methods and datasets)."


### Step 3 — Figure B: Method sensitivity slope chart

Note: this figure was previously named "model sensitivity" in the script
but is now called "method sensitivity" because the comparison is between
methods, not between LLMs. The x-axis (Granite-4 vs Llama-3.3) still
shows model sensitivity per method.

Add in the model analysis paragraph of Section 4 (currently around line 520,
the paragraph starting "Model capabilities prove crucial..."). Insert after
that paragraph:

    \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{chart_method_sensitivity.pdf}
        \caption{%
            Field F1 sensitivity to backbone LLM choice (Granite-4 vs.\
            Llama-3.3-70B) across methods, oracle mode.
            Each line is one method--dataset pair; colour encodes method.
            Dataset labels use abbreviations: ITOps = ITOpsGraph,
            CMS/MIMIC/Synthea = OMOP variants.
            Map-Reflect gains a mean of $+19.6$\,pp with Llama over Granite,
            showing that induced reflection amplifies backbone capability.
            ReAct is almost model-invariant ($+0.2$\,pp mean): its failures
            are structural, not capability-limited.
            MatchMaker shows moderate sensitivity ($+4.7$\,pp mean) with
            one inversion on NL2GQL (Granite 83.2\% $>$ Llama 77.2\%).
        }
        \label{fig:method_sensitivity}
    \end{figure}

Extend the prose anchor sentence from:
    "Model capabilities prove crucial for effective schema mapping."
to:
    "Model capabilities prove crucial for effective schema mapping
     (Figure~\ref{fig:method_sensitivity})."


### Step 4 — Figure C: Signal heatmap (replaces existing ablation figure)

Find the existing figure block (around line 607 in current vldb.tex):

    \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.95\columnwidth]{ablation_oracle.png}
        \caption{Map-Reflect vs MatchMaker on OMOP datasets...}
        \label{fig:ablation}
    \end{figure}

Replace the ENTIRE block with:

    \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{chart_signal_heatmap.pdf}
        \caption{%
            Reflection signal performance across datasets.
            Rows are trigger configurations ordered by ITOpsGraph recall
            (best to worst, top to bottom); columns are datasets
            (ITOps = ITOpsGraph; CMS, MIMIC, Synthea = OMOP variants).
            OMOP values are estimated by applying the ITOpsGraph ablation
            deltas to the oracle Map-Reflect baseline.
            Cell format: value (delta from base).
            The white rule separates the single beneficial configuration
            (UQ only, $+0.2$\,pp) from all configurations that reduce
            accuracy. The red--green colormap makes the performance cliff
            between selective and aggressive reflection immediately legible.
        }
        \label{fig:signal_heatmap}
    \end{figure}

Then update ALL references to the old label:
  Search: \ref{fig:ablation}
  Replace: \ref{fig:signal_heatmap}
  (Affected lines: ~614, ~623 in current vldb.tex)

Also remove ablation_oracle.png from the paper directory since it is
no longer referenced.


## Mean delta values for Figure B caption

Recompute these if the data changes and update the caption accordingly:

    Map-Reflect  mean(Llama - Granite) across datasets = +19.6 pp
    MatchMaker   mean(Llama - Granite) across datasets = +4.7 pp
    ReAct        mean(Llama - Granite) across datasets = +0.2 pp

The MatchMaker NL2GQL inversion (Granite 83.2% > Llama 77.2%) is noted
explicitly in the caption because it is the only case where a stronger
model performs worse.


## Heatmap data fidelity caveat

The OMOP columns in the heatmap are ESTIMATED (ITOpsGraph ablation deltas
applied uniformly). The caption must retain the word "estimated". If the
full ablation is run on OMOP datasets, replace estimated values in the
chart_signal_heatmap() data block and remove "estimated" from the caption.


## Submission checklist

  [ ] chart_api_field_gap.pdf      in paper directory
  [ ] chart_method_sensitivity.pdf in paper directory
  [ ] chart_signal_heatmap.pdf     in paper directory
  [ ] fig:api_field_gap referenced in Section 4 opening
  [ ] fig:method_sensitivity referenced in model analysis paragraph
  [ ] fig:signal_heatmap replaces fig:ablation everywhere
  [ ] ablation_oracle.png removed or no longer referenced
  [ ] No stray \ref{fig:ablation} remaining in document
