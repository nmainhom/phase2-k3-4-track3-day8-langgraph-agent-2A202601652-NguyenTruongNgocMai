# LangGraph Support Ticket Agent Report

## Metrics summary

| Metric | Value |
|---|---:|
| Scenarios | 7 |
| Success rate | 100.0% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Approval interrupts | 2 |
| Checkpoint history available | Yes |

## Scenario results

| Scenario | Result | Expected route | Actual route | Nodes | Retries |
|---|---|---|---|---:|---:|
| S01_simple | Pass | simple | simple | 4 | 0 |
| S02_tool | Pass | tool | tool | 6 | 0 |
| S03_missing | Pass | missing_info | missing_info | 4 | 0 |
| S04_risky | Pass | risky | risky | 8 | 0 |
| S05_error | Pass | error | error | 10 | 2 |
| S06_delete | Pass | risky | risky | 8 | 0 |
| S07_dead_letter | Pass | error | error | 5 | 1 |

## Architecture

The workflow normalizes a ticket, uses structured LLM classification, then routes it to an answer, tool, clarification, or approval path. Tool results are evaluated before a bounded retry loop. Append-only audit events, tool results, and errors make each run observable; a checkpointer can persist state by thread ID.

## Persistence and recovery

The scenario runner uses a SQLite checkpointer and a deterministic `thread_id` per scenario. After processing, it reads checkpoint history for a completed thread; history available: yes.

## Failure analysis

Transient tool errors enter the retry loop and are sent to dead letter once the configured maximum is reached. Risky actions cannot call the tool until the approval node records an approval; rejection requests clarification instead.

## Improvements

Add an interactive approval UI, replace the mock tool with authenticated support APIs, and use an LLM judge with citations to evaluate tool-result quality.
