"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    TODO(student): Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    rows = "\n".join(
        f"| {item.scenario_id} | {'Pass' if item.success else 'Fail'} | {item.expected_route} | "
        f"{item.actual_route or '-'} | {item.nodes_visited} | {item.retry_count} |"
        for item in metrics.scenario_metrics
    )
    return f"""# LangGraph Support Ticket Agent Report

## Metrics summary

| Metric | Value |
|---|---:|
| Scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.1%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Approval interrupts | {metrics.total_interrupts} |

## Scenario results

| Scenario | Result | Expected route | Actual route | Nodes | Retries |
|---|---|---|---|---:|---:|
{rows}

## Architecture

The workflow normalizes a ticket, uses structured LLM classification, then routes it to an answer, tool, clarification, or approval path. Tool results are evaluated before a bounded retry loop. Append-only audit events, tool results, and errors make each run observable; a checkpointer can persist state by thread ID.

## Failure analysis

Transient tool errors enter the retry loop and are sent to dead letter once the configured maximum is reached. Risky actions cannot call the tool until the approval node records an approval; rejection requests clarification instead.

## Improvements

Add an interactive approval UI, replace the mock tool with authenticated support APIs, and use an LLM judge with citations to evaluate tool-result quality.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
