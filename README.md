# Map-Reflect: Uncertainty-Guided Reflection for Schema Mapping

Supplementary software and data for the anonymous submission.

* `software/`: the method (first pass, benefit estimate, selection, reflection strategies, either-order verifier), the
  full-identity scorer, the split-group bootstrap, and the generators of every number, table and chart in the paper.
  See `software/README.md`.
* `data/`: dataset inputs and splits, the frozen configuration fitted on development data, stored model outputs and
  token logs, the facts ledger, and the generated tables. See `data/README.md`.
* `prompts/`: every prompt the paper uses, shown on one worked example (`prompts.pdf`, with its LaTeX source
  `prompts.tex`, generated from the code that sends each prompt).

To score from the stored outputs, merge the two folders into one working directory and run the command in
`data/README.md`:

```
mkdir work && cp -r software/. work/ && rsync -a --exclude README.md data/ work/ && cd work
```

Datasets whose licences do not allow redistribution are not included; `data/README.md` lists them with the converters
that rebuild them.
