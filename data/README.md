# Map-Reflect: supplementary data

Dataset inputs, the frozen configuration, stored model outputs and the generated numbers for "Map-Reflect:
Uncertainty-Guided Reflection for Schema Mapping" (anonymous submission). The code is in `software.zip`. Unzip both
archives into the same directory; the paths below are relative to it and are the paths the code reads.
Both archives carry a README.md at their root, so unzip the second without it:
`unzip software.zip -d supp && unzip data.zip -x README.md -d supp`.

## Contents

| Path | What it is | Paper |
|---|---|---|
| `data/<ds>_dataset.jsonl` (ToolBench: `toolbench_eval_dataset.jsonl`) | Samples: source type, target type, gold field mapping | Sec. 5.1, App. Datasets |
| `data/<ds>_api_lookup.json` | Field descriptions and types for every field of both schemas | Sec. 3.1 |
| `data/<ds>_reflect_lookup_ff1.json`, `data/<ds>_reflect_lookup_v41.json` | The same lookups with description components kept apart, read by the reflection prompts | Sec. 3.4 |
| `data/splits/<ds>.json` | Grouped development and test splits (sample ids and the group of each) | Sec. 5.1 |
| `data/v4_pools/<ds>.json` | Candidate lists shown to the reflection strategies (shown candidates, the rest of the target type, hybrid top 20) | Sec. 3.4 |
| `data/toolbench_dataset.jsonl` | Tool names, read to form ToolBench's split groups | Sec. 5.1 |
| `results/naacl/v2/first/given/<ds>__<model>/*.json` | First-pass outputs: the ranked candidates, scores and stated answer of every field | Sec. 3.1 to 3.2 |
| `results/naacl/v6/v5/{dev,test}/<ds>__<model>.jsonl` | Second-pass outcomes of every field under both reflection strategies, with the verifier's decisions | Sec. 5.3 |
| `results/naacl/v6/v5/tokens/*.jsonl` | Per-call prompt and completion token counts (stage, field key, prompt hash) | cost columns, App. Compute |
| `results/naacl/v4/test/given/<ds>__<model>/fields.jsonl` | Stored outcomes of the reference configuration (Margin-Reflect, method v4.1) | App. Margin-Reflect |
| `docs/naacl/results/v5_frozen.json` | The frozen configuration: selected reflection and verifier, the benefit model (coefficients, standardisation, classes), threshold tau, margin thresholds, fitted on development only | App. Frozen Values |
| `docs/naacl/results/v4_1_frozen.json` | The reference configuration's frozen thresholds | App. Frozen Values |
| `docs/naacl/results/*_facts.json`, `*_frozen.json`, `v5_test.md`, `v5_dev_fit.md`, `v5_cost_diff.md` | Stored facts of each analysis (claim, value, interval, source, generating function, status) | all |
| `docs/naacl/v6/FACTS.md` | The facts ledger: every number in the paper with its identifier and provenance | all |
| `paper/numbers_v6.tex`, `paper/tab6_*.tex`, `paper/figures/data6/*.csv` | The generated number macros, tables and chart data as compiled into the paper | all tables and figures |

Models (`<model>`): `gptoss` (gpt-oss-120b), `llama4` (Llama-4 Maverick), `granite` (Granite-4 small), `qwen35`
(Qwen3.5-397B-A17B). Endpoint names inside stored files are anonymised to ENDPOINT_A and ENDPOINT_B.

Tables the paper inputs, in order of appearance. Main text: `tab6_perdataset`, `tab6_v5_models`,
`tab6_main_compact`, `tab6_family_struct`, `tab6_signals_compact`. Appendix: `tab6_perdataset_granite`, `tab6_main`,
`tab6_d1`, `tab6_evab`, `tab6_signals_ho`, `tab6_signals`, `tab6_xw`, `tab6_compute`, `tab6_main_mr`, `tab6_holm`.
The other `tab6_*.tex` files are generated alongside and not compiled into the paper.

## Datasets included

| Dataset (`<ds>`) | Role | Source | Dev / test samples |
|---|---|---|---|
| ToolBench (`toolbench`) | core | ToolBench RapidAPI instruction files G1 to G3 (Apache 2.0); parameter pairs between tools of one category | 1063 / 1431 |
| NL2GQL (`nl2gql`) | core | NL2GQL corpus, one sample per joined type pair of the gold GraphQL queries | 145 / 193 |
| OMOP-CMS (`omop_cms`) | core | OMAP benchmark (CMS SynPUF to OMOP), as distributed with MatchMaker | 12 / 19 |
| OMOP-Synthea (`omop_synthea`) | core | OMAP benchmark (Synthea to OMOP) | 4 / 7 |
| Valentine ChEMBL, OpenData, TPC-DI, WikiData (`valentine_*`) | core | Valentine suite (Zenodo, CC BY 4.0), as packaged in the LLMatch release | 11/18, 11/19, 11/19, 1/2 |
| OAEI Conference (`oaei_conference`) | held-out | OAEI 2024 Conference track, reference alignment ra1 | 91 / 213 |
| ITOpsGraph (`itopsgraph`) | core | Enterprise IT-operations API schemas (Sec. 4, App. Datasheet); gold labels in `data/labels/itopsgraph_v2.1_cross_product.jsonl` | split in `data/splits/itopsgraph.json` |
| ITOpsGraph crosswalk (`itopsgraph_xw`) | appendix | Crosswalk split of ITOpsGraph (App. Crosswalk); inputs and gold only | split in `data/splits/itopsgraph_xw.json` |

ITOpsGraph's schemas belong to their product owners, whose terms govern redistribution; they are provided for review. All are schema metadata only (type and field names, declared types, descriptions); no data instances and no
personal data.

## Datasets not included

The following sets are used in the paper but their contents are not redistributed here. The converters are in
`software.zip`; each takes the third-party release and writes the evaluation files under `data/`. Their split
manifests (`data/splits/<ds>.json`, sample ids only) are included, so that the converted files can be checked
against the splits the paper used.

| Dataset | Role | Why not included | How to rebuild |
|---|---|---|---|
| OMOP-MIMIC (`omop_mimic`) | core | Derived from MIMIC-III schema documentation; MIMIC material is distributed under credentialed-access terms | OMAP benchmark as distributed with MatchMaker; `scripts/build_omap_dataset.py <omap_dir> omop_mimic` (writes `data/omap_omop_mimic_*`; rename to `data/omop_mimic_*`) |
| MIMIC-OMOP, ReMatch release (`mimic_omop_rematch`) | held-out | MIMIC-III schema documentation (as above) | `scripts/naacl/convert_mimic_omop_rematch.py` on the ReMatch release's `data/` directory |
| SemTab 2024 Metadata2KG round 1 (`semtab_m2kg_r1`) | held-out | Content derives from DBpedia and web tables under their own terms | Zenodo record 14207376, `metadata2kg.zip`; `scripts/naacl/convert_semtab_m2kg.py <round1 dir>` |
| Bio-ML OMIM-ORDO (`bioml_omim_ordo`) | held-out | OMIM class text carries OMIM's own redistribution terms | Zenodo record 13119437, `omim-ordo.zip`; `scripts/naacl/convert_bioml.py <pair dir> omim-ordo` (fixed sample of 500, seeded) |
| SchemaNet CPRD Aurum and GOLD (`schemanet_cprd`) | held-out | The upstream release carries no licence; CPRD schema documentation | LLMatch release `dataset/`; `scripts/naacl/convert_schemanet_cprd.py` |

Stored model outputs for these sets are left out as well, since the first-pass outputs embed the schema text.
Rebuild the inputs, then rerun the first and second pass (see `software.zip`) to score them.

## Scoring from these files

See `software.zip`'s README. In short:

```
ITOPS_V21=data/labels/itopsgraph_v2.1_cross_product.jsonl \
SUPP_DATASETS=itopsgraph,toolbench,nl2gql,omop_cms,omop_synthea,valentine_chembl,valentine_opendata,valentine_tpcdi,valentine_wikidata,oaei_conference \
V5_FROZEN_COMMIT=frozen python scripts/naacl/v5.py --stage score
```

The per-dataset gains it prints equal the per-dataset lines of the stored `docs/naacl/results/v5_test.md`. Pooled
numbers cover only the datasets included here, so they differ from the paper's pooled numbers, which are in
`docs/naacl/results/v5_facts.json` and `docs/naacl/v6/FACTS.md`.

The development fit (`v5.py --stage fit`) needs the development outcomes of all ten core datasets, one of which (OMOP-MIMIC) is
not included, so it cannot be rerun from these files; its result is the stored `v5_frozen.json`.
