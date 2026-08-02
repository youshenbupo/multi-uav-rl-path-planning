# ruff: noqa: E501
"""Build a portable PDF from the frozen AAMAS descriptive summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

if __package__:
    from scripts.build_aamas_descriptive_report import (
        ARM_LABELS,
        CBF_METRICS,
        SCENARIO_SHORT_LABELS,
        SCENARIOS,
        TASK_METRICS,
        _load,
        _paired_rows,
        _three_uav_rows,
    )
else:
    from build_aamas_descriptive_report import (  # type: ignore[no-redef]
        ARM_LABELS,
        CBF_METRICS,
        SCENARIO_SHORT_LABELS,
        SCENARIOS,
        TASK_METRICS,
        _load,
        _paired_rows,
        _three_uav_rows,
    )

TITLE = "Post-isolation multi-UAV descriptive evidence"
PAGE_WIDTH, _PAGE_HEIGHT = landscape(A4)
PLOT_COLORS = ("#1F4E79", "#2A9D8F", "#E9C46A", "#C75C5C")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=HexColor("#17324D"),
            alignment=TA_CENTER,
            spaceAfter=9 * mm,
        ),
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=HexColor("#A23B3B"),
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
        "h1": ParagraphStyle(
            "Heading",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=HexColor("#17324D"),
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=HexColor("#263442"),
            spaceAfter=3 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=HexColor("#415466"),
        ),
    }


def _decorate_page(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setTitle(TITLE)
    canvas.setAuthor("multi-UAV research audit pipeline")
    canvas.setSubject("Descriptive-only AAMAS 2027 evidence package")
    canvas.setStrokeColor(HexColor("#D7E0E8"))
    canvas.line(16 * mm, 13 * mm, PAGE_WIDTH - 16 * mm, 13 * mm)
    canvas.setFillColor(HexColor("#607284"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        16 * mm,
        8.5 * mm,
        "AAMAS 2027 internal evidence package - external submission not authorized",
    )
    canvas.drawRightString(PAGE_WIDTH - 16 * mm, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _three_success_chart(rows: list[dict[str, Any]]) -> Drawing:
    by_cell = {(row["scenario"], row["arm"]): row for row in rows}
    arm_names = list(ARM_LABELS.values())
    values = [
        [float(by_cell[(scenario, arm)]["seed_mean"]) for scenario in SCENARIOS]
        for arm in arm_names
    ]
    drawing = Drawing(720, 285)
    chart = VerticalBarChart()
    chart.x = 55
    chart.y = 45
    chart.height = 190
    chart.width = 615
    chart.data = values
    chart.categoryAxis.categoryNames = [SCENARIO_SHORT_LABELS[item] for item in SCENARIOS]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 1
    chart.valueAxis.valueStep = 0.2
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.barSpacing = 1.5
    chart.groupSpacing = 7
    for index, color in enumerate(PLOT_COLORS):
        chart.bars[index].fillColor = HexColor(color)
        chart.bars[index].strokeColor = None
    drawing.add(chart)
    drawing.add(String(6, 137, "Success fraction", fontName="Helvetica", fontSize=8, angle=90))
    legend = Legend()
    legend.x = 55
    legend.y = 273
    legend.fontName = "Helvetica"
    legend.fontSize = 7
    legend.dx = 8
    legend.dy = 8
    legend.deltax = 150
    legend.columnMaximum = 1
    legend.colorNamePairs = [(HexColor(color), name) for color, name in zip(PLOT_COLORS, arm_names)]
    drawing.add(legend)
    return drawing


def _scale_success_chart(rows: list[dict[str, Any]]) -> Drawing:
    values = [
        [float(row["no_uncertainty_seed_mean"]) for row in rows],
        [float(row["full_seed_mean"]) for row in rows],
    ]
    labels = [str(row["scale_scenario_short"]) for row in rows]
    drawing = Drawing(720, 280)
    chart = VerticalBarChart()
    chart.x = 55
    chart.y = 55
    chart.height = 180
    chart.width = 615
    chart.data = values
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 6.3
    chart.categoryAxis.labels.angle = 35
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 1
    chart.valueAxis.valueStep = 0.2
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = HexColor("#8A94A6")
    chart.bars[0].strokeColor = None
    chart.bars[1].fillColor = HexColor("#1F4E79")
    chart.bars[1].strokeColor = None
    chart.barSpacing = 1.5
    chart.groupSpacing = 5
    drawing.add(chart)
    drawing.add(String(6, 137, "Success fraction", fontName="Helvetica", fontSize=8, angle=90))
    legend = Legend()
    legend.x = 55
    legend.y = 270
    legend.fontName = "Helvetica"
    legend.fontSize = 7
    legend.dx = 8
    legend.dy = 8
    legend.deltax = 150
    legend.columnMaximum = 1
    legend.colorNamePairs = [
        (HexColor("#8A94A6"), "No-uncertainty arm"),
        (HexColor("#1F4E79"), "Full method"),
    ]
    drawing.add(legend)
    return drawing


def _styled_table(data: list[list[Any]], widths: list[float], *, repeat_rows: int = 1) -> LongTable:
    table = LongTable(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#CBD5DF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#F3F6F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _three_success_table(rows: list[dict[str, Any]]) -> LongTable:
    data: list[list[Any]] = [["Scenario", "Method", "Mean", "Seed SD"]]
    data.extend(
        [
            row["scenario_label"],
            row["arm"],
            f"{float(row['seed_mean']):.3f}",
            f"{float(row['seed_sample_std']):.3f}",
        ]
        for row in rows
    )
    return _styled_table(data, [54 * mm, 66 * mm, 28 * mm, 28 * mm])


def _scale_success_table(rows: list[dict[str, Any]]) -> LongTable:
    data: list[list[Any]] = [["Scale / scenario", "No-unc.", "Full", "Delta", "Delta SD"]]
    data.extend(
        [
            row["scale_scenario_label"],
            f"{float(row['no_uncertainty_seed_mean']):.3f}",
            f"{float(row['full_seed_mean']):.3f}",
            f"{float(row['paired_delta_full_minus_no_uncertainty']):+.3f}",
            f"{float(row['paired_delta_sample_std']):.3f}",
        ]
        for row in rows
    )
    return _styled_table(data, [70 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm])


def _metric_inventory(styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [
            Paragraph("Task metrics", styles["small"]),
            Paragraph("CBF diagnostic metrics", styles["small"]),
        ],
        [
            Paragraph("<br/>".join(TASK_METRICS), styles["small"]),
            Paragraph("<br/>".join(CBF_METRICS), styles["small"]),
        ],
    ]
    table = Table(data, colWidths=[115 * mm, 115 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#DCE8F2")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#AEBECD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _story(
    three_success: list[dict[str, Any]],
    scale_success: list[dict[str, Any]],
    *,
    generated_at: str,
    git_revision: str,
) -> list[Any]:
    styles = _styles()
    negative_cells = [
        row for row in scale_success if float(row["paired_delta_full_minus_no_uncertainty"]) < 0
    ]
    negative_text = ", ".join(
        f"{row['scale']} {row['scenario']} ({float(row['paired_delta_full_minus_no_uncertainty']):+.2f})"
        for row in negative_cells
    )
    return [
        Spacer(1, 10 * mm),
        Paragraph("DESCRIPTIVE ONLY - INTERNAL AUDIT SURFACE", styles["eyebrow"]),
        Paragraph(TITLE, styles["title"]),
        Paragraph(
            "Five independently trained seeds per arm, six prespecified scenarios, and 20 episodes per seed-scenario cell. "
            "This PDF is a portable view of the frozen descriptive summaries; the raw JSONL and validated JSON summaries remain authoritative.",
            styles["body"],
        ),
        Paragraph("Evidence boundary", styles["h1"]),
        Paragraph(
            "No superiority, formal safety, component causality, calibrated uncertainty, real-time capability, or robust generalization claim is supported. "
            "The observed directions vary across scale, scenario, and metric, while collision outcomes often share a zero floor.",
            styles["body"],
        ),
        Paragraph("Aggregation contract", styles["h1"]),
        Paragraph(
            "Each seed-scenario value first averages 20 episodes. Across-seed means and sample standard deviations use the five trained seeds as the repetition units. "
            "For 5/8 UAV, every paired delta is full uncertainty-aware minus independently trained no-uncertainty for the same seed.",
            styles["body"],
        ),
        Paragraph("Research mechanism under test", styles["h1"]),
        Paragraph(
            "Communication staleness, delay, and loss -> neighbour-state prediction uncertainty -> uncertainty-aware predictive interaction graph -> coordinated action -> uncertainty-adaptive CBF margin -> dynamic-obstacle execution.",
            styles["body"],
        ),
        Paragraph(f"Generated: {generated_at}<br/>Git revision: {git_revision}", styles["small"]),
        PageBreak(),
        Paragraph("3-UAV success by method and scenario", styles["h1"]),
        Paragraph(
            "Across-seed descriptive means. All four independently trained arms and all six scenarios are shown. The self-loop graph is the historical raw-graph label, not a neighbour-state graph.",
            styles["body"],
        ),
        _three_success_chart(three_success),
        Paragraph(
            "Chart labels: Nom = nominal; Delay = delay only; Loss = loss only; Dyn = dynamic only; Comb = combined; OOD = OOD communication plus obstacle.",
            styles["small"],
        ),
        PageBreak(),
        Paragraph("Exact 3-UAV success evidence", styles["h1"]),
        Paragraph(
            "Mean and sample SD are across the five trained-seed cell means.", styles["body"]
        ),
        _three_success_table(three_success),
        PageBreak(),
        Paragraph("5/8-UAV success by arm, scale, and scenario", styles["h1"]),
        Paragraph(
            "The grouped bars show both independently trained arm means at 5 and 8 UAV. They provide cell context and are not an inferential effect plot. "
            f"The exact paired table retains unfavorable cells, including {negative_text}.",
            styles["body"],
        ),
        _scale_success_chart(scale_success),
        PageBreak(),
        Paragraph("5/8-UAV paired success evidence", styles["h1"]),
        Paragraph(
            "Both arm means, full-minus-no-uncertainty delta, and paired-delta sample SD are retained for every scale-scenario cell. "
            "The ablation jointly removes graph-risk and CBF-margin uncertainty and therefore cannot identify a component effect.",
            styles["body"],
        ),
        _scale_success_table(scale_success),
        PageBreak(),
        Paragraph("Complete metric and provenance contract", styles["h1"]),
        Paragraph(
            "The source summaries retain all 11 task metrics and five CBF diagnostic metrics for every eligible cell. "
            "Task outcomes and solver diagnostics remain separate; timing-sensitive replay results cannot replace the raw fallback ledger.",
            styles["body"],
        ),
        _metric_inventory(styles),
        Spacer(1, 5 * mm),
        Paragraph("Limitations", styles["h1"]),
        Paragraph(
            "Only five trained seeds are available and no inferential procedure was fixed before values were inspected. The uncertainty bound is deterministic rather than calibrated. "
            "The online CBF uses centralized simulator geometry, slack, discrete integration, and emergency fallback. Current evidence is therefore descriptive simulation evidence only.",
            styles["body"],
        ),
        Paragraph("Traceability", styles["h1"]),
        Paragraph(
            "The adjacent provenance JSON records exact input paths and SHA-256 hashes, PDF hash, page count, Git revision, scenario set, visible row counts, and claim boundary. "
            "Eligibility and exclusions remain governed by docs/aamas2027_analysis_plan.md and docs/post_actor_isolation_descriptive_validation.md.",
            styles["body"],
        ),
    ]


def build_pdf_report(
    three_uav_path: Path,
    five_uav_path: Path,
    eight_uav_path: Path,
    output_pdf: Path,
    output_provenance: Path,
    *,
    generated_at: str,
    git_revision: str,
) -> dict[str, Any]:
    """Build the report and return its validation metadata."""
    if output_pdf.exists() or output_provenance.exists():
        raise FileExistsError("PDF report outputs already exist; refusing to overwrite.")
    if output_pdf.parent != output_provenance.parent:
        raise ValueError("PDF and provenance outputs must share one unique output root.")

    three = _three_uav_rows(_load(three_uav_path))
    five = _paired_rows(_load(five_uav_path), "5 UAV")
    eight = _paired_rows(_load(eight_uav_path), "8 UAV")
    three_success = three["success"]
    scale_success = five["success"] + eight["success"]

    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title=TITLE,
        author="multi-UAV research audit pipeline",
        subject="Descriptive-only AAMAS 2027 evidence package",
        pageCompression=1,
        invariant=1,
    )
    document.build(
        _story(
            three_success,
            scale_success,
            generated_at=generated_at,
            git_revision=git_revision,
        ),
        onFirstPage=_decorate_page,
        onLaterPages=_decorate_page,
    )

    reader = PdfReader(output_pdf)
    page_count = len(reader.pages)
    if page_count < 4:
        raise ValueError("Generated PDF unexpectedly lacks the required report sections.")
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required_text = (
        "DESCRIPTIVE ONLY",
        "3-UAV success by method and scenario",
        "5/8-UAV paired success evidence",
        "OOD comm. + obstacle",
    )
    missing_text = [item for item in required_text if item not in extracted_text]
    if missing_text:
        raise ValueError(f"Generated PDF failed text-integrity checks: {missing_text}")

    provenance = {
        "generated_at": generated_at,
        "git_revision": git_revision,
        "builder": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "pdf_path": output_pdf.as_posix(),
        "pdf_sha256": _sha256(output_pdf),
        "page_count": page_count,
        "inputs": {
            "three_uav": {"path": three_uav_path.as_posix(), "sha256": _sha256(three_uav_path)},
            "five_uav": {"path": five_uav_path.as_posix(), "sha256": _sha256(five_uav_path)},
            "eight_uav": {"path": eight_uav_path.as_posix(), "sha256": _sha256(eight_uav_path)},
        },
        "independent_unit": "trained_seed",
        "trained_seed_count": 5,
        "episodes_per_seed_scenario": 20,
        "scenarios": list(SCENARIOS),
        "visible_success_rows": {
            "three_uav": len(three_success),
            "scale_paired": len(scale_success),
        },
        "charts": {
            "three_uav_success": {"series": list(ARM_LABELS)},
            "scale_success": {
                "series": ["no_uncertainty_seed_mean", "full_seed_mean"]
            },
        },
        "retained_task_metrics": list(TASK_METRICS),
        "retained_cbf_diagnostic_metrics": list(CBF_METRICS),
        "paired_delta_direction": "full_minus_independently_trained_no_uncertainty",
        "claim_boundary": "descriptive_only",
        "external_submission_authorized": False,
    }
    output_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-uav-summary", type=Path, required=True)
    parser.add_argument("--five-uav-summary", type=Path, required=True)
    parser.add_argument("--eight-uav-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--git-revision", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("PDF output directory already exists; refusing to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build_pdf_report(
        args.three_uav_summary,
        args.five_uav_summary,
        args.eight_uav_summary,
        args.output_dir / "aamas2027_descriptive_report.pdf",
        args.output_dir / "provenance.json",
        generated_at=args.generated_at,
        git_revision=args.git_revision,
    )


if __name__ == "__main__":
    main()
