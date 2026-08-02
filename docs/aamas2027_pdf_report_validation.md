# AAMAS 2027 portable PDF report validation

**Validation date:** 2026-08-02

**Decision:** the `retry2` PDF is the current eligible portable descriptive
evidence surface. It is an internal research artifact, not an external
submission, and it does not make inferential, superiority, safety, causal, or
robust-generalization claims.

## Eligible artifact

- PDF:
  `output/pdf/aamas2027_descriptive_report_20260802_retry2/aamas2027_descriptive_report.pdf`
- PDF SHA-256:
  `C37B1C931C9031D89FAAC568428800BB1AC264ECC96C821D742F1F92C25A6F09`
- provenance sidecar:
  `output/pdf/aamas2027_descriptive_report_20260802_retry2/provenance.json`
- provenance SHA-256:
  `96149F4BA7815A43B64FAB9511F1221E0855559A06E06A2D37B0E7BACF163AFA`;
- page count: 6 landscape A4 pages;
- generator:
  `scripts/build_aamas_descriptive_pdf.py`;
- generator SHA-256 at generation:
  `E86B4459C3CB90A8639334E46425815174E249579269028C62E0904B85C8CDD3`;
- parent Git revision recorded by the sidecar:
  `a3100971129ba6c347e1407a24b5cc2f0780b000`.

The generator hash is recorded separately because the generator and validation
documents were still pending commit when the artifact was produced.
Repository `.gitattributes` marks `*.pdf` as binary; the staged Git blobs match
the raw working-tree blobs, preventing line-ending conversion from invalidating
the recorded PDF hashes.

## Authoritative inputs

The PDF reads the three already frozen and independently validated descriptive
summaries without recomputing raw-episode aggregates:

1. 3-UAV four-arm summary:
   `B3CB5F017A241DAF29C366DA9592624076A2FE44519F51809D5AE50251CF2F3E`;
2. 5-UAV paired summary:
   `6661DBDDF678039581ED7FEC0D84595A641C25B87C6877DCEA95C3ADD5359120`;
3. 8-UAV paired summary:
   `E487CD7C16EAA285025917848AD51268BDC11228174ED17EFB395DC740144C61`.

Eligibility, raw-root checks, all 4,800 episode records and aggregate
recomputation remain governed by
`docs/post_actor_isolation_descriptive_validation.md`. The PDF is a display
surface, not a replacement for that audit or the raw JSONL.

## Visible content contract

The six pages retain:

- the descriptive-only and no-external-submission boundary;
- the fixed communication-to-uncertainty-to-graph-to-CBF mechanism chain;
- a four-series 3-UAV success chart over all six scenarios;
- an exact 24-row 3-UAV success table;
- a two-series 5/8-UAV chart showing both independently trained arm means over
  all 12 scale-scenario cells;
- an exact 12-row paired success table containing both means, full-minus-
  no-uncertainty delta, and paired-delta sample standard deviation;
- the complete inventory of 11 task and five CBF diagnostic metrics retained
  by the source summaries; and
- limitations and traceability statements.

The scale chart deliberately compares the two nonnegative arm means rather
than plotting a signed delta. Exact signed deltas remain in the adjacent table.
No unfavorable, zero, or positive scale-scenario cell is removed.

## Automated validation

The generator reopens the PDF with `pypdf`, verifies at least four pages and
requires the descriptive boundary, 3-UAV title, 5/8-UAV paired title and OOD
label in extracted text before writing provenance.

Independent post-generation checks established:

- `%PDF-` magic and valid `pypdf`/`pdfplumber` parsing;
- six pages, no encryption, no JavaScript, no form and no attachment;
- one page size only: 841.89 by 595.276 points;
- PDF SHA-256 equals the sidecar value;
- generator SHA-256 equals the sidecar value;
- all six scenarios, 11 task metrics and five CBF metrics are recorded;
- required claim-boundary text is extractable;
- page 3 contains one 25-row extracted table including its header;
- page 5 contains one 13-row extracted table including its header; and
- `external_submission_authorized` is false.

`pdfinfo` independently reports six landscape A4 pages, PDF 1.4, no encryption,
no JavaScript and no form. Poppler rendered all pages at 144 DPI. It emitted
missing-display-font diagnostics for `Symbol` and `ArialUnicode`, but the
complete page inspection found no missing characters, black boxes, clipping,
overlap or illegible labels.

## Visual validation

All six rendered pages were inspected at original detail:

1. title, evidence boundary, aggregation rule and mechanism chain are complete;
2. the four-arm 3-UAV chart and legend are readable;
3. all 24 3-UAV rows fit without clipping;
4. the paired scale chart shows the three nonzero no-uncertainty cells and the
   zero full-method cells without implying an inferential test;
5. all 12 paired rows and signed deltas fit without clipping; and
6. metric inventory, limitations, traceability, footer and page number are
   readable.

No visual defect remains in the eligible `retry2` PDF.

Repository verification passes Ruff, mypy over 105 source files and six
focused PDF/report/protocol tests. The full suite is 165 passed with the same
one external-MATLAB-inventory failure and 190 OSQP dependency warnings; no RL,
report or PDF test fails.

## Preserved excluded attempts

- `output/pdf/aamas2027_descriptive_report_20260802_launcher_aborted/` records
  a zero-artifact direct-script import failure. The CLI was then repaired under
  a failing regression test.
- `output/pdf/aamas2027_descriptive_report_20260802_retry1/` retains the first
  six-page PDF and provenance plus `ABORTED.json`. It rendered cleanly, but its
  scale plot contained only the all-zero full-method series and was judged
  visually uninformative. It is excluded rather than overwritten.

The nine earlier HTML report roots remain separately preserved and excluded.
The validated PDF resolves the portable evidence-surface gate; it does not make
the failed HTML eligible and does not remove the need for manuscript-specific
figure/table traceability or the final manuscript-to-artifact audit.
