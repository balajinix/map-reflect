# Method v5 on test (scored once)

13,384 field-model pairs. Frozen at 463f9c40d6f2d752a4ec5b95bafb57e182feeba1 2026-10-11T01:16:35+05:30. Every v5 policy uses the same measured second pass. Intervals: split-group bootstrap, B = 2000.

| Policy | accuracy | gain over keep | minus v4.1 | minus struct | routed | calls per field | tokens per field | repairs | breaks |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Keep the first answer | 85.49 | +0.00 [+0.00, +0.00] | -5.41 [-8.62, -3.00] | -7.08 [-11.34, -3.70] | 0.0% | 0.00 | 0 | 0 | 0 |
| Map-Reflect v5 (benefit router) | 93.05 | +7.56 [+4.22, +11.88] | +2.15 [+1.12, +3.34] | +0.48 [+0.15, +0.90] | 32.0% | 0.44 | 1,835 | 1,026 | 14 |
| v5 intervention, margin router (t_m refitted) | 93.16 | +7.67 [+4.34, +11.94] | +2.26 [+1.23, +3.40] | +0.58 [+0.20, +1.08] | 56.3% | 0.69 | 2,137 | 1,042 | 16 |
| v5 intervention, margin router at the v4.1 thresholds | 92.98 | +7.49 [+4.20, +11.70] | +2.08 [+1.12, +3.20] | +0.41 [+0.05, +0.87] | 42.7% | 0.54 | 1,836 | 1,017 | 14 |
| v5 intervention, struct router | 92.57 | +7.08 [+3.70, +11.34] | +1.67 [+0.60, +2.92] |  | 26.7% | 0.37 | 1,838 | 953 | 5 |
| v5 intervention, benefit router or struct | 93.05 | +7.56 [+4.22, +11.88] | +2.15 [+1.12, +3.34] | +0.48 [+0.15, +0.90] | 36.7% | 0.49 | 2,115 | 1,026 | 14 |
| v5 intervention on every field | 93.10 | +7.61 [+4.25, +11.92] | +2.20 [+1.18, +3.38] | +0.53 [+0.15, +1.03] | 100.0% | 1.13 | 3,060 | 1,048 | 29 |
| v5 intervention, random at v5's share | 87.95 | +2.46 [+1.37, +3.85] | -2.95 [-4.82, -1.55] | -4.62 [-7.51, -2.32] | 32.0% | 0.36 | 982 | 339 | 9 |
| Oracle at v5's share | 93.32 | +7.83 [+4.50, +12.09] | +2.42 [+1.40, +3.55] | +0.75 [+0.37, +1.25] |  |  |  | 1,048 | 0 |
| Method v4.1 (stored outcomes, reference) | 90.90 | +5.41 [+3.00, +8.62] |  | -1.67 [-2.92, -0.60] |  |  |  | 745 | 21 |

Per model and per dataset, gain over keep (v5 / v4.1 / struct / every field), points:

- gptoss (3,346): +6.72 / +4.81 / +5.95 / +6.93
- llama4 (3,346): +7.86 / +5.17 / +7.20 / +7.92
- granite (3,346): +4.81 / +2.15 / +4.45 / +4.84
- qwen35 (3,346): +10.85 / +9.50 / +10.73 / +10.76
- itopsgraph (1,508): +0.27 / +0.27 / +0.20 / +0.27
- toolbench (6,180): +1.52 / +1.41 / +1.26 / +1.52
- nl2gql (772): +0.78 / +0.65 / +0.65 / +1.55
- omop_cms (340): +0.88 / +0.88 / +0.29 / +0.88
- omop_mimic (312): +1.28 / +0.64 / +1.28 / +1.28
- omop_synthea (260): +1.54 / +1.92 / +0.38 / +1.54
- valentine_chembl (920): +17.28 / +7.17 / +13.59 / +17.39
- valentine_opendata (2,172): +31.26 / +23.53 / +31.26 / +31.26
- valentine_tpcdi (864): +7.18 / +4.63 / +5.90 / +7.18
- valentine_wikidata (56): -5.36 / +1.79 / +1.79 / -5.36

## Held-out sets (3,652 field-model pairs; the core-development configuration transferred)

| Policy | accuracy | gain over keep | minus v4.1 | calls per field |
|---|--:|--:|--:|--:|
| Keep the first answer | 71.85 | +0.00 [+0.00, +0.00] | -2.49 [-3.26, -1.78] | 0.00 |
| Map-Reflect v5 (benefit router) | 74.75 | +2.90 [+1.95, +3.99] | +0.41 [-0.16, +1.04] | 1.03 |
| v5 intervention, margin router at the v4.1 thresholds | 74.64 | +2.79 [+1.87, +3.85] | +0.30 [-0.24, +0.91] | 0.95 |
| v5 intervention, struct router | 74.59 | +2.74 [+1.79, +3.79] | +0.25 [-0.33, +0.89] | 0.92 |
| v5 intervention on every field | 74.73 | +2.88 [+1.92, +3.98] | +0.38 [-0.22, +1.03] | 1.15 |
| Method v4.1 (stored outcomes, reference) | 74.34 | +2.49 [+1.78, +3.26] |  |  |
- mimic_omop_rematch (432): v5 +6.48, v4.1 +3.01
- semtab_m2kg_r1 (564): v5 +6.21, v4.1 +6.03
- oaei_conference (852): v5 +1.53, v4.1 +1.17
- bioml_omim_ordo (1,336): v5 +2.32, v4.1 +2.47
- schemanet_cprd (468): v5 -0.21, v4.1 +0.21
