"""Generate an HTML visualization for ARC benchmark result JSON files.

This script scans the ``results`` directory and converts every ``.json`` file into
an interactive HTML view. Each task is rendered as a section containing every
attempt, along with metadata and a color-coded grid for answers expressed as
2D integer arrays.

Usage::

    python visualize_results.py [--results-dir path/to/results] [--output results.html]

Both flags are optional – by default the script looks for ``results`` relative to
the project root (the file's parent directory) and writes ``results/index.html``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# Fixed ARC color palette (0-9). Remaining values fall back to a soft purple.
ARC_COLORS = {
    0: "#000000",  # black
    1: "#2D7DF6",  # vivid blue
    2: "#F24C43",  # bright red
    3: "#5CC031",  # lime green
    4: "#F7D433",  # yellow
    5: "#9C9FA7",  # neutral gray
    6: "#EB4DBF",  # magenta
    7: "#F88B29",  # orange
    8: "#7EC9F4",  # sky blue
    9: "#8E1F3D",  # maroon
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create HTML visualisations for ARC results JSON files")
    parser.add_argument(
        "--results-dir",
        default=Path(__file__).resolve().parent / "submissions",
        type=Path,
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output HTML file (defaults to <results-dir>/index.html)",
    )
    return parser.parse_args()


def find_json_files(results_dir: Path) -> List[Path]:
    """Return all JSON files under ``results_dir`` sorted for stable output."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    return sorted(results_dir.rglob("*.json"))


def value_to_color(value: Optional[int]) -> str:
    """Map ARC cell values to colors."""
    return ARC_COLORS.get(value, "#6C5CE7")


def render_grid(grid: List[List[int]]) -> str:
    """Render a 2D integer grid as a responsive CSS grid."""
    if not grid:
        return "<div class=\"grid empty\">No grid data</div>"

    num_cols = max((len(row) for row in grid if isinstance(row, list)), default=0)
    if num_cols == 0:
        return "<div class=\"grid empty\">No grid data</div>"

    cells: List[str] = []
    for row in grid:
        for value in row:
            color = value_to_color(value if isinstance(value, int) else None)
            value_str = sanitize_html(str(value))
            cells.append(
                (
                    "<div class=\"cell\" style=\"background-color: {color}\" "
                    "data-cell-value=\"{value}\"></div>"
                ).format(color=color, value=value_str)
            )

    style = f"--grid-cols: {num_cols};"
    return "<div class=\"grid\" style=\"{style}\">{cells}</div>".format(
        style=style,
        cells="".join(cells),
    )


def pretty_json(data: object) -> str:
    """Return a pretty formatted JSON string for display."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)


def compute_duration_seconds(start: Optional[str], end: Optional[str]) -> Optional[float]:
    """Compute duration from ISO timestamps, returning seconds if possible."""
    if not start or not end:
        return None

    try:
        start_dt = dt.datetime.fromisoformat(start)
        end_dt = dt.datetime.fromisoformat(end)
    except ValueError:
        return None

    duration = (end_dt - start_dt).total_seconds()
    return duration if duration >= 0 else None


def render_prompt_context(
    prompt_context_data: List[Tuple[str, str, List[List[int]]]]
) -> str:
    """Render grouped prompt context grids extracted from the user message."""

    if not prompt_context_data:
        return ""

    grouped = {}
    for example_label, grid_label, grid in prompt_context_data:
        example_name = example_label.strip("- ") if example_label else "Prompt"
        category_key = "other"
        label_upper = (grid_label or "").upper()
        if "INPUT" in label_upper:
            category_key = "inputs"
        elif "OUTPUT" in label_upper:
            category_key = "outputs"

        entry = grouped.setdefault(
            example_name,
            {"inputs": [], "outputs": [], "other": []},
        )
        entry[category_key].append((grid_label, grid))

    example_sections: List[str] = []
    for example_name, data in grouped.items():
        input_blocks = data["inputs"]
        output_blocks = data["outputs"]
        paired_rows: List[str] = []

        for input_block, output_block in zip_longest(input_blocks, output_blocks):
            cells: List[str] = []
            if input_block:
                label, grid = input_block
                caption = sanitize_html(label) if label else "Input"
                cells.append(
                    "<div class=\"context-grid-block\"><div class=\"context-grid-caption\">{}</div>{}</div>".format(
                        caption,
                        render_grid(grid),
                    )
                )
            if output_block:
                label, grid = output_block
                caption = sanitize_html(label) if label else "Output"
                cells.append(
                    "<div class=\"context-grid-block\"><div class=\"context-grid-caption\">{}</div>{}</div>".format(
                        caption,
                        render_grid(grid),
                    )
                )
            if cells:
                paired_rows.append(
                    "<div class=\"context-pair\">{}</div>".format("".join(cells))
                )

        other_blocks = data["other"]
        other_html = ""
        if other_blocks:
            other_cells = []
            for label, grid in other_blocks:
                caption = sanitize_html(label) if label else "Context"
                other_cells.append(
                    "<div class=\"context-grid-block\"><div class=\"context-grid-caption\">{}</div>{}</div>".format(
                        caption,
                        render_grid(grid),
                    )
                )
            other_html = "<div class=\"context-other\"><h6>Additional</h6><div class=\"context-pair\">{}</div></div>".format(
                "".join(other_cells)
            )

        content = "".join(paired_rows) + other_html
        if not content:
            content = "<p class=\"no-context\">No grids</p>"

        example_sections.append(
            "<div class=\"context-example\"><h5>{name}</h5>{content}</div>".format(
                name=sanitize_html(example_name),
                content=content,
            )
        )

    if not example_sections:
        return ""

    return "<details class=\"prompt-context\"><summary>Prompt Context</summary><div class=\"context-example-list\">{}</div></details>".format(
        "".join(example_sections)
    )


def render_message_section(title: str, entries: List[dict]) -> str:
    if not entries:
        return ""

    blocks: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        label = entry.get("label")
        heading_parts = [f"{title} {idx}" if len(entries) > 1 else title]
        if label:
            heading_parts.append(label)
        heading = " – ".join(part for part in heading_parts if part)
        content = entry.get("content", "")
        grids = entry.get("grids", [])

        block_html = "<pre>{}</pre>".format(sanitize_html(str(content)))

        if grids:
            grid_blocks = []
            for example_label, grid_label, grid in grids:
                caption_parts = []
                if example_label:
                    caption_parts.append(example_label.strip("- "))
                if grid_label:
                    caption_parts.append(grid_label)
                caption = " • ".join(caption_parts) if caption_parts else "Grid"
                grid_blocks.append(
                    "<div class=\"prompt-grid\"><div class=\"prompt-grid-label\">{}</div>{}</div>".format(
                        sanitize_html(caption),
                        render_grid(grid),
                    )
                )
            block_html += "<div class=\"prompt-grid-wrapper\">{}</div>".format("".join(grid_blocks))

        blocks.append(
            "<div class=\"message-block\"><h5>{heading}</h5>{html}</div>".format(
                heading=sanitize_html(heading),
                html=block_html,
            )
        )

    return "<details class=\"message-section\"><summary>{title}</summary><div class=\"message-block-list\">{blocks}</div></details>".format(
        title=sanitize_html(title),
        blocks="".join(blocks),
    )


def extract_prompt_grids(prompt: str) -> List[Tuple[str, str, List[List[int]]]]:
    """Best-effort extraction of 2D integer arrays from a prompt string.

    Returns a list of tuples ``(example_label, grid_label, grid)`` where
    ``example_label`` is the most recent line that looks like an example header
    (e.g. ``--Example 0--``) prior to the grid, and ``grid_label`` is the line
    immediately preceding the grid (e.g. ``INPUT:`` or ``OUTPUT:``).
    """

    grids: List[Tuple[str, str, List[List[int]]]] = []
    idx = 0

    while True:
        start = prompt.find("[[", idx)
        if start == -1:
            break

        depth = 0
        end = start
        while end < len(prompt):
            char = prompt[end]
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1

        if depth != 0:
            break

        snippet = prompt[start:end]
        idx = end

        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue

        if not (isinstance(parsed, list) and parsed and all(isinstance(r, list) for r in parsed)):
            continue

        preceding_text = prompt[:start]
        lines = preceding_text.splitlines()
        grid_label = ""
        example_label = ""

        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if not grid_label:
                grid_label = stripped
            if stripped.startswith("--") and stripped.endswith("--"):
                example_label = stripped
                break

        grids.append((example_label, grid_label, parsed))

    return grids


def sanitize_html(text: str) -> str:
    """Simple HTML escaping for text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_attempt(attempt_name: str, attempt: dict) -> Tuple[str, List[Tuple[str, str, List[List[int]]]]]:
    """Render a single attempt block."""
    if not attempt:
        return f"<div class=\"attempt empty\"><h4>{sanitize_html(attempt_name)}</h4><p>No data</p></div>"

    answer = attempt.get("answer")
    metadata = attempt.get("metadata", {})
    correct = attempt.get("correct")

    grid_html = ""
    if isinstance(answer, list):
        grid_html = render_grid(answer)
    elif answer is None:
        grid_html = "<p class=\"no-answer\">No answer</p>"
    else:
        grid_html = "<pre class=\"raw-answer\">{}</pre>".format(sanitize_html(str(answer)))

    tags: List[str] = []
    provider = metadata.get("provider")
    model_name = metadata.get("model")
    if provider or model_name:
        tags.append(
            "<span class=\"tag\">{}</span>".format(
                sanitize_html(" / ".join(bit for bit in [provider, model_name] if bit))
            )
        )

    usage = metadata.get("usage") or {}
    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        tags.append(f"<span class=\"tag\">{sanitize_html(str(total_tokens))} tokens</span>")

    cost_info = metadata.get("cost") or {}
    total_cost = cost_info.get("total_cost")
    if total_cost is not None:
        try:
            cost_display = f"${float(total_cost):.4f}"
        except (TypeError, ValueError):
            cost_display = str(total_cost)
        tags.append(
            "<span class=\"tag\">cost {}</span>".format(
                sanitize_html(cost_display)
            )
        )

    duration = compute_duration_seconds(metadata.get("start_timestamp"), metadata.get("end_timestamp"))
    if duration is not None:
        tags.append(f"<span class=\"tag\">{duration:.2f}s</span>")

    correct_label = ""
    if isinstance(correct, bool):
        status = "correct" if correct else "incorrect"
        correct_label = f"<span class=\"badge {status}\">{status.title()}</span>"
    else:
        correct_label = "<span class=\"badge unknown\">Unknown</span>"

    details_sections: List[str] = []

    prompt_context_data: List[Tuple[str, str, List[List[int]]]] = []

    if usage or cost_info:
        usage_rows: List[str] = []
        if usage:
            usage_rows.append(
                "<tr><th>Total tokens</th><td>{}</td></tr>".format(
                    sanitize_html(str(total_tokens)) if total_tokens is not None else "–"
                )
            )
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if value is not None:
                    usage_rows.append(
                        "<tr><th>{label}</th><td>{value}</td></tr>".format(
                            label=sanitize_html(key.replace("_", " ").title()),
                            value=sanitize_html(str(value)),
                        )
                    )

            completion_details = usage.get("completion_tokens_details") or {}
            if completion_details:
                for detail_key, detail_value in completion_details.items():
                    usage_rows.append(
                        "<tr><th>{label}</th><td>{value}</td></tr>".format(
                            label=sanitize_html(detail_key.replace("_", " ").title()),
                            value=sanitize_html(str(detail_value)),
                        )
                    )

        if cost_info:
            for key in ("prompt_cost", "completion_cost", "reasoning_cost", "total_cost"):
                value = cost_info.get(key)
                if value is None:
                    continue
                try:
                    display = f"${float(value):.4f}"
                except (TypeError, ValueError):
                    display = sanitize_html(str(value))
                usage_rows.append(
                    "<tr><th>{label}</th><td>{value}</td></tr>".format(
                        label=sanitize_html(key.replace("_", " ").title()),
                        value=sanitize_html(display),
                    )
                )

        usage_table = "".join(usage_rows)
        details_sections.append(
            "<details><summary>Usage & Cost</summary><table class=\"metrics-table\">{rows}</table></details>".format(
                rows=usage_table or "<tr><td>No usage data</td></tr>"
            )
        )

    reasoning = metadata.get("reasoning_summary")
    if reasoning:
        details_sections.append(
            "<details><summary>Reasoning Summary</summary><pre>{}</pre></details>".format(
                sanitize_html(reasoning)
            )
        )

    choices = metadata.get("choices") or []
    user_entries: List[dict] = []
    assistant_entries: List[dict] = []
    other_entries: List[dict] = []

    for choice in choices:
        index = choice.get("index")
        message = choice.get("message") or {}
        role = (message.get("role") or "message").lower()
        content = str(message.get("content", ""))
        finish_reason = choice.get("finish_reason")
        grids = extract_prompt_grids(content)
        label_parts = []
        if index is not None:
            label_parts.append(f"Choice {index}")
        if finish_reason:
            label_parts.append(f"finish: {finish_reason}")
        label = " – ".join(label_parts)

        entry = {
            "label": label,
            "content": content,
            "grids": grids,
        }

        if role == "user":
            user_entries.append(entry)
            if not prompt_context_data:
                prompt_context_data = grids
        elif role == "assistant":
            assistant_entries.append(entry)
        else:
            other_entries.append(entry)

    input_section = render_message_section("Model Input", user_entries)
    output_section = render_message_section("Model Output", assistant_entries)
    other_section = render_message_section("Other Messages", other_entries)

    for section in (input_section, output_section, other_section):
        if section:
            details_sections.append(section)

    if metadata:
        details_sections.append(
            "<details><summary>Metadata (raw)</summary><pre>{}</pre></details>".format(
                sanitize_html(pretty_json(metadata))
            )
        )

    details_sections.append(
        "<details><summary>Answer (raw)</summary><pre>{}</pre></details>".format(
            sanitize_html(pretty_json(answer))
        )
    )

    tags_html = "".join(tags)
    details_html = "".join(details_sections)

    return (
        "<div class=\"attempt\">"
        "<div class=\"attempt-header\"><h4>{name} {status}</h4><div class=\"tag-list\">{tags}</div></div>"
        "<div class=\"attempt-body\">{grid}{details}</div>"
        "</div>"
    ).format(
        name=sanitize_html(attempt_name),
        status=correct_label,
        tags=tags_html,
        grid=grid_html,
        details=details_html,
    ), prompt_context_data


def render_task(task_path: Path, task_data: Iterable[dict], base_dir: Path) -> str:
    """Render a task file (each JSON file)."""
    sections: List[str] = []
    for index, pair_attempts in enumerate(task_data, start=1):
        pair_header = f"<h3>Test Pair {index}</h3>"
        attempts_html: List[str] = []
        prompt_context: List[Tuple[str, str, List[List[int]]]] = []
        if isinstance(pair_attempts, dict):
            for key in sorted(pair_attempts.keys()):
                if not key.startswith("attempt_"):
                    continue
                attempt_html, context_data = render_attempt(key, pair_attempts.get(key))
                if not prompt_context and context_data:
                    prompt_context = context_data
                attempts_html.append(attempt_html)
        else:
            attempts_html.append("<p class=\"no-attempts\">Unexpected format</p>")

        prompt_context_html = render_prompt_context(prompt_context)

        sections.append(
            "<section class=\"test-pair\">{header}{context}<div class=\"attempt-list\">{attempts}</div></section>".format(
                header=pair_header,
                context=prompt_context_html,
                attempts="".join(attempts_html) or "<p class=\"no-attempts\">No attempts</p>",
            )
        )

    file_title = sanitize_html(task_path.stem)
    try:
        relative_path = sanitize_html(str(task_path.relative_to(base_dir)))
    except ValueError:
        relative_path = sanitize_html(str(task_path))
    return (
        "<article class=\"task\" id=\"{file_title}\">"
        "<h2>{file_title}</h2><div class=\"file-path\">{relative_path}</div>"
        "{sections}</article>"
    ).format(file_title=file_title, relative_path=relative_path, sections="".join(sections))


def build_html(json_files: List[Path], base_dir: Path) -> str:
    generated_at = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    task_sections: List[str] = []

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            error_block = (
                f"<article class=\"task error\" id=\"{sanitize_html(path.stem)}\">"
                f"<h2>{sanitize_html(path.stem)}</h2>"
                f"<p class=\"error\">Failed to parse JSON: {sanitize_html(str(exc))}</p>"
                "</article>"
            )
            task_sections.append(error_block)
            continue

        if not isinstance(data, list):
            error_block = (
                f"<article class=\"task error\" id=\"{sanitize_html(path.stem)}\">"
                f"<h2>{sanitize_html(path.stem)}</h2>"
                "<p class=\"error\">Unexpected JSON root structure (expected list)</p></article>"
            )
            task_sections.append(error_block)
            continue

        task_sections.append(render_task(path, data, base_dir=base_dir))

    css = """
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f4f6fb; color: #222; }
header { background: #1f2937; color: #fff; padding: 1.5rem 2rem; }
header h1 { margin: 0; font-size: 1.75rem; }
header p { margin: 0.5rem 0 0; opacity: 0.85; }
main { padding: 2rem; max-width: 1200px; margin: 0 auto; }
.task { background: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }
.task.error { border: 2px solid #f87171; }
.task h2 { margin-top: 0; }
.task .file-path { font-size: 0.9rem; color: #6b7280; margin-bottom: 0.75rem; }
.test-pair { margin-top: 1.5rem; }
.test-pair h3 { margin-bottom: 0.75rem; color: #111827; }
.attempt-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.attempt { background: #f9fafb; border-radius: 10px; padding: 1rem; border: 1px solid #e5e7eb; display: flex; flex-direction: column; gap: 0.75rem; }
.attempt-header { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; justify-content: space-between; }
.attempt-header h4 { margin: 0; display: flex; align-items: center; gap: 0.5rem; font-size: 1rem; }
.tag-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.tag { background: #e5e7eb; color: #111827; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.attempt.empty { color: #6b7280; }
.attempt-body { display: flex; flex-direction: column; gap: 0.75rem; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.badge.correct { background: #bbf7d0; color: #166534; }
.badge.incorrect { background: #fecaca; color: #991b1b; }
.badge.unknown { background: #e5e7eb; color: #1f2937; }
.grid { --grid-cols: 1; --cell-size: 24px; display: inline-grid; grid-template-columns: repeat(var(--grid-cols), 1fr); grid-auto-rows: 1fr; width: min(calc(var(--grid-cols) * var(--cell-size)), 100%); border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
.grid .cell { width: 100%; aspect-ratio: 1 / 1; }
.grid.empty { padding: 0.5rem; font-size: 0.85rem; color: #6b7280; }
.no-answer, .no-attempts, .error { color: #b91c1c; font-size: 0.9rem; }
details { margin-top: 0.5rem; }
details summary { cursor: pointer; color: #2563eb; }
pre { background: #111827; color: #e5e7eb; padding: 0.75rem; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; }
.metrics-table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; }
.metrics-table th, .metrics-table td { text-align: left; padding: 0.25rem 0.4rem; font-size: 0.85rem; }
.metrics-table th { width: 40%; color: #374151; font-weight: 600; }
.metrics-table tr:nth-child(odd) { background: rgba(229, 231, 235, 0.35); }
.message-section { margin-top: 0.25rem; }
.message-block-list { display: flex; flex-direction: column; gap: 1rem; margin-top: 0.75rem; }
.message-block h5 { margin: 0 0 0.5rem; font-size: 0.9rem; color: #1f2937; }
.prompt-grid-wrapper { display: grid; gap: 0.75rem; margin-top: 0.75rem; }
.prompt-grid { background: #fff; border: 1px solid #d1d5db; padding: 0.5rem; border-radius: 8px; }
.prompt-grid-label { font-size: 0.75rem; font-weight: 600; color: #4b5563; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.04em; }
.prompt-context { margin-top: 0.25rem; }
.context-example-list { display: flex; flex-direction: column; gap: 1.25rem; margin-top: 0.75rem; }
.context-example h5 { margin: 0 0 0.5rem; font-size: 0.95rem; color: #111827; }
.context-pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-bottom: 0.5rem; }
.context-grid-block { background: #fff; border: 1px solid #d1d5db; padding: 0.5rem; border-radius: 8px; display: flex; flex-direction: column; gap: 0.4rem; }
.context-grid-caption { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #4b5563; font-weight: 600; }
.context-other { margin-top: 0.5rem; }
.context-other h6 { margin: 0 0 0.35rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; }
.no-context { font-size: 0.85rem; color: #6b7280; }
@media (max-width: 640px) { main { padding: 1rem; } .attempt-list { grid-template-columns: 1fr; } }
"""

    html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>ARC Results Visualisation</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>ARC Benchmark Results</h1>
    <p>Generated {generated_at}</p>
  </header>
  <main>
    {sections}
  </main>
</body>
</html>
""".format(css=css, generated_at=generated_at, sections="".join(task_sections))

    return html


def write_html(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    output_path: Path = args.output or (results_dir / "index.html")

    json_files = find_json_files(results_dir)
    if not json_files:
        raise SystemExit(f"No JSON files found in {results_dir}")

    html = build_html(json_files, base_dir=results_dir)
    write_html(output_path, html)
    print(f"Wrote visualisation to {output_path}")


if __name__ == "__main__":
    main()
