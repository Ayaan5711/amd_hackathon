"""Report sections for the Survey Analytics pack.

Six sections, each built the same way as the governance report's sections
(`app/packs/governance/report.py`): a prompt-builder reconstructs the survey
DataFrame from the investigation's entries (`reconstruct_df_and_schema`), runs the
relevant pure-pandas / existing survey tool (segment stats, trend comparison, anomaly
flagging) to compute real numbers, and pairs the prompt with a content-aware mock
fabricator so `_report_node`'s single `call_llm_async` per section produces
data-grounded markdown in LLM_MODE=mock.

`extract_open_text_themes` and `recommend_actions` (app/tools/*) each make their own
internal LLM call, so the themes/recommendations sections deliberately do NOT call
them - they compute the same kind of context via pandas instead, keeping this module
to one LLM call per section.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from app.packs.survey.categorical import (
    build_demographic_profile,
    build_response_summary,
    build_segment_response_crosstab,
    split_demographic_and_response_columns,
)
from app.packs.survey.common import (
    pick_dimension_column,
    pick_excluded_columns,
    pick_metric_columns,
    pick_segment_column,
    reconstruct_df_and_schema,
)
from app.tools.anomaly_flag import flag_anomalies
from app.tools.segment_stats import get_segment_stats
from app.tools.trend_compare import compare_trends
from app.utils.csv_loader import get_categorical_columns, get_column_by_type, get_numeric_columns
from app.utils.llm_client import MockFabricator

SURVEY_REPORT_PROMPT = """You are a people-analytics consultant authoring the "{section_title}" \
section of a survey insights report.

SURVEY OVERVIEW:
{summary}

{extra}

Write the "{section_title}" section as concise markdown (use headings and bullet points, and \
reference specific segments/columns as evidence where relevant).

Respond with JSON only, no other text:
{{"content": "<markdown text for this section>"}}
"""


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------


def _segment_table(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    """get_segment_stats for every numeric column, grouped by the primary segment column."""
    categorical_cols = get_categorical_columns(schema)
    numeric_cols = get_numeric_columns(schema)
    dimension_col = pick_dimension_column(categorical_cols)
    segment_col = pick_segment_column(categorical_cols, exclude=dimension_col)
    if not segment_col or not numeric_cols:
        return None

    results = [
        r
        for r in (
            get_segment_stats(df, schema, segment_column=segment_col, metric_column=metric_col)
            for metric_col in numeric_cols
        )
        if r["success"]
    ]
    if not results:
        return None
    return {"segment_column": segment_col, "results": results}


def _segment_table_markdown(segment_column: str, results: list[dict[str, Any]]) -> str:
    segment_names = [s["segment"] for s in results[0]["segments"]]
    header = f"| Metric | {' | '.join(segment_names)} | Best | Worst | Gap |"
    sep = "|" + "---|" * (len(segment_names) + 4)
    rows = [header, sep]
    for result in results:
        means = {s["segment"]: s["mean"] for s in result["segments"]}
        cells = " | ".join(_fmt(means.get(name)) for name in segment_names)
        best = f"{result['best_segment']} ({_fmt(means.get(result['best_segment']))})"
        worst = f"{result['worst_segment']} ({_fmt(means.get(result['worst_segment']))})"
        rows.append(f"| {result['metric_column']} | {cells} | {best} | {worst} | {result['gap']} |")
    return "\n".join(rows)


def _segment_highlight(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    table = _segment_table(df, schema)
    return table["results"][0] if table else None


def _segment_mean(result: dict[str, Any], segment_name: str) -> float | None:
    for s in result["segments"]:
        if s["segment"] == segment_name:
            return s["mean"]
    return None


def _anomaly_report(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    numeric_cols = get_numeric_columns(schema)
    if not numeric_cols:
        return None
    result = flag_anomalies(df, schema, columns=numeric_cols)
    return result if result["success"] else None


def _trend_report(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    categorical_cols = get_categorical_columns(schema)
    numeric_cols = get_numeric_columns(schema)
    dimension_col = pick_dimension_column(categorical_cols)
    if not dimension_col or not numeric_cols:
        return None
    result = compare_trends(df, schema, dimension_column=dimension_col, metric_columns=pick_metric_columns(numeric_cols))
    return result if result["success"] else None


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in",
    "on", "for", "with", "this", "that", "it", "at", "as", "be", "has", "have", "had",
    "i", "we", "our", "my", "your", "their", "more", "very", "not", "no", "so", "than",
    "then", "there", "these", "those", "from", "by", "about", "into", "over", "also",
    "just", "really", "felt", "would", "could", "should", "all", "any", "some", "most",
    "much", "many", "use", "used", "using", "new", "next", "out", "been", "things",
    "feel", "feels", "feeling", "quarter", "lately", "lot", "still",
}

_POSITIVE_WORDS = {
    "great", "good", "happy", "enjoy", "enjoyed", "enjoying", "helpful", "smooth",
    "appreciate", "appreciated", "improve", "improved", "improving", "manageable",
    "flexible", "support", "supportive", "decent", "positive", "love", "loved",
    "excellent", "help", "helped", "helps",
}

_NEGATIVE_WORDS = {
    "confusing", "confused", "tight", "isolating", "isolated", "difficult", "outdated",
    "concern", "concerns", "concerning", "need", "needs", "unclear", "poor", "unhappy",
    "frustrating", "frustrated", "bad", "worse", "worst", "lacking", "stress", "stressed",
}

_WORD_PATTERN = re.compile(r"[a-zA-Z']+")


def _keyword_themes(df: pd.DataFrame, schema: dict[str, Any], max_themes: int = 5) -> dict[str, Any] | None:
    """Pure-pandas keyword frequency + lexicon-based sentiment over an open-text column."""
    text_cols = get_column_by_type(schema, "open_text")
    if not text_cols:
        return None

    text_col = text_cols[0]
    responses = df[text_col].dropna().astype(str)
    responses = responses[responses.str.len() > 3]
    if responses.empty:
        return None

    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    pos_hits = 0
    neg_hits = 0
    for resp in responses:
        words = set(_WORD_PATTERN.findall(resp.lower()))
        for word in words:
            if len(word) > 3 and word not in _STOPWORDS:
                counter[word] += 1
                examples.setdefault(word, resp)
        if words & _POSITIVE_WORDS:
            pos_hits += 1
        if words & _NEGATIVE_WORDS:
            neg_hits += 1

    total = len(responses)
    neutral_hits = max(total - pos_hits - neg_hits, 0)
    top_keywords = [
        {"keyword": word, "count": count, "percent": round(count / total * 100, 1), "example": examples[word]}
        for word, count in counter.most_common(max_themes)
    ]

    return {
        "text_column": text_col,
        "total_responses": total,
        "top_keywords": top_keywords,
        "sentiment": {
            "positive_percent": round(pos_hits / total * 100, 1),
            "negative_percent": round(neg_hits / total * 100, 1),
            "neutral_percent": round(neutral_hits / total * 100, 1),
        },
    }


def _recommendation_context(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    numeric_cols = get_numeric_columns(schema)
    if not numeric_cols:
        return None

    key_metrics = []
    for col in numeric_cols[:5]:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        key_metrics.append({
            "name": col,
            "mean": round(float(series.mean()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        })
    if not key_metrics:
        return None
    lowest_metrics = sorted(key_metrics, key=lambda x: x["mean"])[:3]

    categorical_cols = get_categorical_columns(schema)
    dimension_col = pick_dimension_column(categorical_cols)
    segment_col = pick_segment_column(categorical_cols, exclude=dimension_col)

    segment_insights = []
    if segment_col:
        for metric in lowest_metrics:
            result = get_segment_stats(df, schema, segment_column=segment_col, metric_column=metric["name"])
            if result["success"]:
                segment_insights.append(result)

    return {
        "total_responses": len(df),
        "key_metrics": key_metrics,
        "lowest_metrics": lowest_metrics,
        "segment_column": segment_col,
        "segment_insights": segment_insights,
    }


def _demographic_and_response_columns(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Categorical columns split into (demographic_cols, response_cols), excluding
    whatever the numeric-segment path (`_segment_table`) already claims."""
    categorical_cols = get_categorical_columns(schema)
    exclude = pick_excluded_columns(categorical_cols)
    return split_demographic_and_response_columns(df, schema, categorical_cols, exclude=exclude)


def _demographic_context(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Demographic profile + response distribution + a couple of cross-tabs, for the
    Executive Summary's "Demographic Profile" / "Key Findings" / "Recommended
    Actions" blocks. `None` if the dataset has neither demographic nor response
    columns (e.g. `survey_df`, `sample_survey.csv`)."""
    demographic_cols, response_cols = _demographic_and_response_columns(df, schema)
    if not demographic_cols and not response_cols:
        return None

    profile = build_demographic_profile(df, schema, demographic_cols) if demographic_cols else None
    if profile is not None and not profile["success"]:
        profile = None

    response_summary = build_response_summary(df, schema, response_cols) if response_cols else None
    if response_summary is not None and not response_summary["success"]:
        response_summary = None

    crosstabs = []
    if demographic_cols and response_cols:
        crosstabs.append(build_segment_response_crosstab(df, schema, demographic_cols[0], response_cols[0]))
        if len(demographic_cols) > 1:
            crosstabs.append(build_segment_response_crosstab(df, schema, demographic_cols[1], response_cols[0]))
    crosstabs = [c for c in crosstabs if c["success"]]

    return {"profile": profile, "response_summary": response_summary, "crosstabs": crosstabs}


def _crosstab_finding_sentence(crosstab: dict[str, Any], response_summary: dict[str, Any] | None) -> str | None:
    """A "Men show materially higher concern than women: 64.4% vs 32.1%"-style
    sentence for one cross-tab, comparing the segments with the highest and lowest
    share of the response column's dominant value."""
    response_column = crosstab["response_column"]
    dominant_value = crosstab["options"][0] if crosstab["options"] else None
    if response_summary:
        for q in response_summary["questions"]:
            if q["column"] == response_column:
                dominant_value = q["dominant_value"]
                break
    if dominant_value is None:
        return None

    segment_percents = [
        (seg["segment"], next((d["percent"] for d in seg["distribution"] if d["value"] == dominant_value), 0.0))
        for seg in crosstab["segments"]
    ]
    if not segment_percents:
        return None

    if len(segment_percents) == 1:
        segment, percent = segment_percents[0]
        return f"Among **{segment}** respondents, **{_pct(percent)}** selected **{dominant_value}** for **{response_column}**."

    segment_percents.sort(key=lambda item: item[1], reverse=True)
    best_segment, best_pct = segment_percents[0]
    worst_segment, worst_pct = segment_percents[-1]
    return (
        f"**{best_segment}** respondents show materially higher **{response_column}** "
        f"'{dominant_value}' share than **{worst_segment}** ({_pct(best_pct)} vs {_pct(worst_pct)})."
    )


def _demographic_recommended_actions(demo_ctx: dict[str, Any]) -> list[str]:
    """1-3 action bullets for the Executive Summary's "Recommended Actions" block,
    derived from the cross-tab with the highest-percentage segment, falling back to
    the globally-dominant response value, then a generic monitoring action."""
    response_summary = demo_ctx["response_summary"]
    actions: list[str] = []

    for crosstab in demo_ctx["crosstabs"][:2]:
        response_column = crosstab["response_column"]
        dominant_value = crosstab["options"][0] if crosstab["options"] else None
        if response_summary:
            for q in response_summary["questions"]:
                if q["column"] == response_column:
                    dominant_value = q["dominant_value"]
                    break
        if dominant_value is None:
            continue

        best_segment, best_pct = max(
            (
                (seg["segment"], next((d["percent"] for d in seg["distribution"] if d["value"] == dominant_value), 0.0))
                for seg in crosstab["segments"]
            ),
            key=lambda item: item[1],
        )
        actions.append(
            f"Prioritize outreach in **{best_segment}**, where **{_pct(best_pct)}** expect "
            f"**{dominant_value}** for **{response_column}** - the highest of any **{crosstab['segment_column']}**."
        )

    if not actions and response_summary and response_summary["questions"]:
        q = response_summary["questions"][0]
        actions.append(
            f"Investigate the drivers behind the **{_pct(q['dominant_percent'])}** of respondents "
            f"expecting **{q['dominant_value']}** for **{q['column']}**."
        )

    return actions[:3] if actions else ["Continue monitoring demographic and response distributions for emerging shifts."]


def _response_table(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    """`build_response_summary` over the dataset's "response" columns, or `None` if
    there are none (e.g. `survey_df`, `sample_survey.csv`)."""
    _, response_cols = _demographic_and_response_columns(df, schema)
    if not response_cols:
        return None
    result = build_response_summary(df, schema, response_cols)
    return result if result["success"] else None


def _response_table_markdown(result: dict[str, Any]) -> str:
    questions = result["questions"]
    all_options: list[str] = []
    for q in questions:
        for opt in q["options"]:
            if opt not in all_options:
                all_options.append(opt)

    header = f"| Question | {' | '.join(all_options)} | Dominant | Dominant % |"
    sep = "|" + "---|" * (len(all_options) + 3)
    rows = [header, sep]
    for q in questions:
        percents = {d["value"]: d["percent"] for d in q["distribution"]}
        cells = " | ".join(_pct(percents.get(opt, 0.0)) for opt in all_options)
        rows.append(f"| {q['column']} | {cells} | {q['dominant_value']} | {_pct(q['dominant_percent'])} |")
    return "\n".join(rows)


def _demographic_crosstab_tables(df: pd.DataFrame, schema: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Up to 3 demographic columns x first 2 response columns of cross-tabs (the
    "Full demographic analysis" tables), or `None` if there are no
    demographic/response columns to cross."""
    demographic_cols, response_cols = _demographic_and_response_columns(df, schema)
    if not demographic_cols or not response_cols:
        return None

    tables = [
        result
        for segment_col in demographic_cols[:3]
        for response_col in response_cols[:2]
        for result in (build_segment_response_crosstab(df, schema, segment_col, response_col),)
        if result["success"]
    ]
    return tables or None


def _crosstab_markdown(crosstab: dict[str, Any]) -> str:
    options = crosstab["options"]
    header = f"| {crosstab['segment_column']} | {' | '.join(options)} | Dominant | Dominant % |"
    sep = "|" + "---|" * (len(options) + 3)
    rows = [header, sep]
    for seg in crosstab["segments"]:
        percents = {d["value"]: d["percent"] for d in seg["distribution"]}
        cells = " | ".join(_pct(percents.get(opt, 0.0)) for opt in options)
        rows.append(f"| {seg['segment']} | {cells} | {seg['dominant_value']} | {_pct(seg['dominant_percent'])} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Section 1: Executive Summary
# ---------------------------------------------------------------------------


def _mock_executive_summary(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        dashboard = ctx["dashboard"]
        dist = dashboard["risk_distribution"]

        lines = [
            "# Executive Summary",
            "",
            f"This run analyzed **{ctx['total_entries']}** survey responses. "
            f"**{ctx['total_flagged']}** were dispatched for follow-up review based on automated "
            f"triage (PII scan + statistical outlier detection).",
            "",
            "## Risk Distribution",
            f"- Critical: {dist.get('critical', 0)}",
            f"- High: {dist.get('high', 0)}",
            f"- Medium: {dist.get('medium', 0)}",
            f"- Low: {dist.get('low', 0)}",
            "",
        ]

        segment = _segment_highlight(df, schema)
        lines.append("## Segment Highlight")
        if segment:
            lines.append(
                f"`{segment['metric_column']}` by `{segment['segment_column']}`: best segment "
                f"**{segment['best_segment']}** vs. worst **{segment['worst_segment']}** "
                f"(gap of {segment['gap']})."
            )
        else:
            lines.append("No segment with enough responses for a reliable comparison.")
        lines.append("")

        anomalies = _anomaly_report(df, schema)
        lines.append("## Outliers")
        if anomalies and anomalies["columns_with_anomalies"]:
            lines.append(anomalies["summary"])
        else:
            lines.append("No statistically significant outliers detected.")
        lines.append("")

        pii_flagged = dashboard["findings_by_category"]["pii"]["flagged"]
        lines.append("## PII")
        if pii_flagged:
            lines.append(
                f"{pii_flagged} response(s) contain detectable PII in open-text fields - "
                f"review before sharing this dataset externally."
            )
        else:
            lines.append("No PII detected in open-text fields.")

        demo_ctx = _demographic_context(df, schema)
        if demo_ctx is not None:
            if demo_ctx["profile"]:
                lines += ["", "## Demographic Profile"]
                for p in demo_ctx["profile"]["profiles"]:
                    lines.append(f"- **{p['column']}**: {p['top_value']} ({_pct(p['top_percent'])})")

            if demo_ctx["response_summary"] or demo_ctx["crosstabs"]:
                lines += ["", "## Key Findings"]
                if demo_ctx["response_summary"]:
                    for q in demo_ctx["response_summary"]["questions"][:5]:
                        lines.append(
                            f"- **{_pct(q['dominant_percent'])}** of respondents selected "
                            f"**{q['dominant_value']}** for **{q['column']}**."
                        )
                for crosstab in demo_ctx["crosstabs"][:2]:
                    sentence = _crosstab_finding_sentence(crosstab, demo_ctx["response_summary"])
                    if sentence:
                        lines.append(f"- {sentence}")

            lines += ["", "## Recommended Actions"]
            for i, action in enumerate(_demographic_recommended_actions(demo_ctx), start=1):
                lines.append(f"{i}. {action}")

        return {"content": "\n".join(lines)}

    return fabricator


def build_executive_summary_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    dashboard = ctx["dashboard"]
    segment = _segment_highlight(df, schema)
    anomalies = _anomaly_report(df, schema)

    summary_lines = [
        f"- Total responses: {ctx['total_entries']}",
        f"- Flagged for follow-up: {ctx['total_flagged']}",
        f"- Risk distribution: {dashboard['risk_distribution']}",
        f"- PII findings: {dashboard['findings_by_category']['pii']['flagged']}",
    ]
    extra_lines = []
    if segment:
        extra_lines.append(
            f"Segment stats: {segment['metric_column']} by {segment['segment_column']} - "
            f"best={segment['best_segment']}, worst={segment['worst_segment']}, gap={segment['gap']}"
        )
    if anomalies:
        extra_lines.append(f"Anomaly summary: {anomalies['summary']}")

    demo_ctx = _demographic_context(df, schema)
    if demo_ctx is not None:
        if demo_ctx["profile"]:
            for p in demo_ctx["profile"]["profiles"]:
                extra_lines.append(f"Demographic: {p['column']} top value={p['top_value']} ({p['top_percent']}%)")
        if demo_ctx["response_summary"]:
            for q in demo_ctx["response_summary"]["questions"]:
                extra_lines.append(f"Response: {q['column']} dominant={q['dominant_value']} ({q['dominant_percent']}%)")
        for crosstab in demo_ctx["crosstabs"]:
            extra_lines.append(f"Crosstab: {crosstab['response_column']} by {crosstab['segment_column']}: {crosstab['segments']}")

    prompt = SURVEY_REPORT_PROMPT.format(
        section_title="Executive Summary",
        summary="\n".join(summary_lines),
        extra="\n".join(extra_lines) if extra_lines else "(no additional segment/outlier data available)",
    )
    return prompt, _mock_executive_summary(ctx)


# ---------------------------------------------------------------------------
# Section 2: Segment Analysis
# ---------------------------------------------------------------------------


def _mock_segment_analysis(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        table = _segment_table(df, schema)

        lines = ["# Segment Analysis", ""]
        if table is None:
            lines.append(
                "No categorical column with enough responses per group (and at least one "
                "numeric metric) was found for a segment breakdown."
            )
        else:
            results = table["results"]
            lines.append(
                f"Responses were grouped by **{table['segment_column']}** and compared across "
                f"{len(results)} numeric metric(s)."
            )
            lines.append("")
            lines.append(_segment_table_markdown(table["segment_column"], results))
            lines.append("")

            largest = max(results, key=lambda r: r["gap"])
            best_mean = _segment_mean(largest, largest["best_segment"])
            worst_mean = _segment_mean(largest, largest["worst_segment"])
            lines.append(
                f"The widest gap is in **{largest['metric_column']}**: **{largest['best_segment']}** "
                f"({_fmt(best_mean)}) outperforms **{largest['worst_segment']}** ({_fmt(worst_mean)}) "
                f"by {largest['gap']} points."
            )

        response_table = _response_table(df, schema)
        if response_table:
            lines += ["", "## Response Distribution", "", _response_table_markdown(response_table)]

        crosstabs = _demographic_crosstab_tables(df, schema)
        if crosstabs:
            lines += ["", "## Full Demographic Analysis"]
            for crosstab in crosstabs:
                lines += ["", f"### {crosstab['response_column']} by {crosstab['segment_column']}", "", _crosstab_markdown(crosstab)]

        return {"content": "\n".join(lines)}

    return fabricator


def build_segment_analysis_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    table = _segment_table(df, schema)

    if table is None:
        summary = "No categorical segment column with enough responses per group was found."
        extra_parts = ["(no segment data available)"]
    else:
        results = table["results"]
        summary_lines = [
            f"- Segment column: {table['segment_column']}",
            f"- Metrics compared: {[r['metric_column'] for r in results]}",
        ]
        for r in results:
            summary_lines.append(f"  - {r['metric_column']}: best={r['best_segment']}, worst={r['worst_segment']}, gap={r['gap']}")
        summary = "\n".join(summary_lines)
        extra_parts = [_segment_table_markdown(table["segment_column"], results)]

    response_table = _response_table(df, schema)
    if response_table:
        extra_parts.append("Response distribution:\n" + _response_table_markdown(response_table))

    crosstabs = _demographic_crosstab_tables(df, schema)
    if crosstabs:
        for crosstab in crosstabs:
            extra_parts.append(f"{crosstab['response_column']} by {crosstab['segment_column']}:\n" + _crosstab_markdown(crosstab))

    extra = "\n\n".join(extra_parts)

    prompt = SURVEY_REPORT_PROMPT.format(section_title="Segment Analysis", summary=summary, extra=extra)
    return prompt, _mock_segment_analysis(ctx)


# ---------------------------------------------------------------------------
# Section 3: Trends Analysis
# ---------------------------------------------------------------------------


def _mock_trends_analysis(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        result = _trend_report(df, schema)
        if result is None:
            return {
                "content": (
                    "# Trends Analysis\n\n"
                    "No time-based dimension (e.g. a 'Quarter' or 'Period' column) was found in "
                    "this dataset, so no trend comparison is available."
                )
            }

        metrics = result["metric_columns"]
        lines = [
            "# Trends Analysis",
            "",
            f"Comparing {', '.join(metrics)} across **{result['dimension_column']}** "
            f"({result['period_count']} periods: {', '.join(result['periods'])}).",
            "",
            f"| {result['dimension_column']} | Count | " + " | ".join(metrics) + " |",
            "|" + "---|" * (len(metrics) + 2),
        ]
        for row in result["data"]:
            cells = " | ".join(_fmt(row.get(f"{m}_mean")) for m in metrics)
            lines.append(f"| {row['dimension_value']} | {row['count']} | {cells} |")
        lines.append("")

        if result["significant_changes"]:
            lines.append("## Notable Changes")
            for ch in result["significant_changes"]:
                direction = "increased" if ch["change"] > 0 else "decreased"
                pct = f" ({ch['pct_change']}%)" if ch.get("pct_change") is not None else ""
                lines.append(f"- **{ch['metric']}** {direction} by {abs(ch['change'])}{pct} from {ch['period']}.")
        else:
            lines.append("No period-over-period change of 0.5 or more was detected.")

        return {"content": "\n".join(lines)}

    return fabricator


def build_trends_analysis_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    result = _trend_report(df, schema)

    if result is None:
        summary = "No time-based dimension column (e.g. 'Quarter' or 'Period') was found in this dataset."
        extra = "(no trend data available)"
    else:
        summary = "\n".join([
            f"- Dimension: {result['dimension_column']} ({result['period_count']} periods: {result['periods']})",
            f"- Metrics: {result['metric_columns']}",
        ])
        extra_lines = ["Period averages:"]
        for row in result["data"]:
            extra_lines.append(f"- {row}")
        if result["significant_changes"]:
            extra_lines.append("Significant changes:")
            for ch in result["significant_changes"]:
                extra_lines.append(f"- {ch}")
        extra = "\n".join(extra_lines)

    prompt = SURVEY_REPORT_PROMPT.format(section_title="Trends Analysis", summary=summary, extra=extra)
    return prompt, _mock_trends_analysis(ctx)


# ---------------------------------------------------------------------------
# Section 4: Themes & Sentiment
# ---------------------------------------------------------------------------


def _mock_themes_and_sentiment(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        themes = _keyword_themes(df, schema)
        if themes is None:
            return {"content": "# Themes & Sentiment\n\nNo open-text responses were found in this dataset."}

        lines = [
            "# Themes & Sentiment",
            "",
            f"Analyzed **{themes['total_responses']}** open-text response(s) in `{themes['text_column']}`.",
            "",
            "## Top Keywords",
        ]
        for kw in themes["top_keywords"]:
            lines.append(f"- **{kw['keyword']}** - {kw['count']} response(s) ({kw['percent']}%). e.g. \"{kw['example']}\"")

        lines += [
            "",
            "## Overall Sentiment",
            f"- Positive: {themes['sentiment']['positive_percent']}%",
            f"- Negative: {themes['sentiment']['negative_percent']}%",
            f"- Neutral: {themes['sentiment']['neutral_percent']}%",
        ]
        return {"content": "\n".join(lines)}

    return fabricator


def build_themes_and_sentiment_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    themes = _keyword_themes(df, schema)

    if themes is None:
        summary = "No open-text column was found in this dataset."
        extra = "(no theme data available)"
    else:
        summary = "\n".join([
            f"- Open-text column: {themes['text_column']}",
            f"- Responses analyzed: {themes['total_responses']}",
            f"- Sentiment split: {themes['sentiment']}",
        ])
        extra_lines = ["Top keywords with example quotes:"]
        for kw in themes["top_keywords"]:
            extra_lines.append(f"- {kw['keyword']} ({kw['count']} mentions, {kw['percent']}%): \"{kw['example']}\"")
        extra = "\n".join(extra_lines)

    prompt = SURVEY_REPORT_PROMPT.format(section_title="Themes & Sentiment", summary=summary, extra=extra)
    return prompt, _mock_themes_and_sentiment(ctx)


# ---------------------------------------------------------------------------
# Section 5: Anomalies & Data Quality
# ---------------------------------------------------------------------------


def _mock_anomalies_and_quality(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        result = _anomaly_report(df, schema)
        if result is None:
            return {"content": "# Anomalies & Data Quality\n\nNo numeric columns were available for anomaly detection."}

        lines = ["# Anomalies & Data Quality", "", result["summary"], ""]
        if result["anomalies_by_column"]:
            lines.append("## Outliers by Column")
            for a in result["anomalies_by_column"]:
                lines.append(
                    f"- **{a['column']}**: mean={a['mean']}, std={a['std']}, "
                    f"{a['outlier_count']} outlier(s) ({a['outlier_percent']}% of responses, "
                    f"z > {a['z_threshold']})."
                )
                for ev in a["extreme_values"]:
                    lines.append(f"  - value {ev['value']} ({ev['direction']}, z={ev['z_score']})")
        else:
            lines.append("No columns had statistically significant outliers.")

        if result.get("data_quality_issues"):
            lines += ["", "## Data Quality Issues"]
            for issue in result["data_quality_issues"]:
                lines.append(f"- **{issue['column']}**: {issue['issue']} - {issue['details']}")

        return {"content": "\n".join(lines)}

    return fabricator


def build_anomalies_and_quality_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    result = _anomaly_report(df, schema)

    if result is None:
        summary = "No numeric columns were available for anomaly detection."
        extra = "(no anomaly data available)"
    else:
        summary = result["summary"]
        extra_lines = [
            f"Columns analyzed: {result['columns_analyzed']}",
            f"Total anomalies found: {result['total_anomalies_found']}",
        ]
        for a in result["anomalies_by_column"]:
            extra_lines.append(
                f"- {a['column']}: {a['outlier_count']} outlier(s) ({a['outlier_percent']}%), extremes={a['extreme_values']}"
            )
        if result.get("data_quality_issues"):
            extra_lines.append(f"Data quality issues: {result['data_quality_issues']}")
        extra = "\n".join(extra_lines)

    prompt = SURVEY_REPORT_PROMPT.format(section_title="Anomalies & Data Quality", summary=summary, extra=extra)
    return prompt, _mock_anomalies_and_quality(ctx)


# ---------------------------------------------------------------------------
# Section 6: Recommendations
# ---------------------------------------------------------------------------


def _mock_recommendations(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = reconstruct_df_and_schema(ctx["entries"])
        rec_ctx = _recommendation_context(df, schema)
        if rec_ctx is None:
            return {"content": "# Recommendations\n\nNo numeric metrics were available to generate recommendations."}

        lines = [
            "# Recommendations",
            "",
            f"Based on {rec_ctx['total_responses']} response(s), the following actions are "
            f"prioritized by current performance gaps:",
            "",
        ]
        for i, metric in enumerate(rec_ctx["lowest_metrics"], start=1):
            lines.append(
                f"{i}. **Improve {metric['name']}** - currently averaging {metric['mean']} "
                f"(range {metric['min']}-{metric['max']}). Investigate root causes and set a "
                f"target above the current average."
            )

        if rec_ctx["segment_insights"]:
            lines += ["", "## Segment-Specific Focus"]
            for seg in rec_ctx["segment_insights"]:
                worst_mean = _segment_mean(seg, seg["worst_segment"])
                best_mean = _segment_mean(seg, seg["best_segment"])
                lines.append(
                    f"- For **{seg['metric_column']}**, prioritize **{seg['worst_segment']}** "
                    f"(avg {worst_mean}) - it trails **{seg['best_segment']}** (avg {best_mean}) "
                    f"by {seg['gap']} points."
                )

        lines += [
            "",
            "## Summary",
            f"Top opportunity areas: {', '.join(m['name'] for m in rec_ctx['lowest_metrics'])}.",
        ]
        return {"content": "\n".join(lines)}

    return fabricator


def build_recommendations_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = reconstruct_df_and_schema(ctx["entries"])
    rec_ctx = _recommendation_context(df, schema)

    if rec_ctx is None:
        summary = "No numeric metrics were available in this dataset."
        extra = "(no recommendation data available)"
    else:
        summary = "\n".join([
            f"- Total responses: {rec_ctx['total_responses']}",
            f"- Lowest-performing metrics: {rec_ctx['lowest_metrics']}",
        ])
        if rec_ctx["segment_insights"]:
            extra_lines = [f"Segment column: {rec_ctx['segment_column']}", "Segment gaps:"]
            for seg in rec_ctx["segment_insights"]:
                extra_lines.append(
                    f"- {seg['metric_column']}: best={seg['best_segment']}, worst={seg['worst_segment']}, gap={seg['gap']}"
                )
            extra = "\n".join(extra_lines)
        else:
            extra = "(no segment gap data available)"

    prompt = SURVEY_REPORT_PROMPT.format(section_title="Recommendations", summary=summary, extra=extra)
    return prompt, _mock_recommendations(ctx)


# Ordered: section_name -> prompt builder. Iteration order is report order.
SURVEY_REPORT_SECTIONS: dict[str, Any] = {
    "executive_summary": build_executive_summary_prompt,
    "segment_analysis": build_segment_analysis_prompt,
    "trends_analysis": build_trends_analysis_prompt,
    "themes_and_sentiment": build_themes_and_sentiment_prompt,
    "anomalies_and_quality": build_anomalies_and_quality_prompt,
    "recommendations": build_recommendations_prompt,
}
