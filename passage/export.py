"""Turn trace rows into fine-tuning and translation-memory files.

    python -m passage.export --format dpo --in data/traces --out dpo.jsonl
    python -m passage.export --format sft --in ./bucket-dump --out sft.jsonl
    python -m passage.export --format tmx --in data/traces --out memory.tmx

`--in` takes any number of directories, read recursively for *.jsonl: the local
trace directory, a `gcloud storage cp -r` dump of the contribution bucket, or
both. Rows are joined per (trace, segment):

- **sft**: one chat example per segment a human settled on, meaning either
  approved as the model wrote it or edited into its final form.
- **dpo**: one preference pair per edited segment. The model's output is
  `rejected` and the human's final text is `chosen`. Edits are the only ground
  truth this app gets (traces.py), so this is the most valuable export.
- **tmx**: TMX 1.4 translation memory (source to final text) for CAT tools.

A segment the model wrote and nobody looked at is not exported: an untouched
machine translation is not evidence of anything. Pairs a human edited are also
the defensible core for training (OpenAI's terms restrict using raw outputs to
build competing models).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


@dataclass
class Segment:
    trace_id: str
    segment_id: str
    target_language: str = ""
    engine: str = ""
    source: str = ""
    machine: str = ""
    edits: list[tuple[float, str]] = field(default_factory=list)
    approved: bool | None = None
    session: str | None = None

    @property
    def final(self) -> str | None:
        """What the human settled on, or None if nobody looked."""
        if self.edits:
            return sorted(self.edits)[-1][1]
        if self.approved:
            return self.machine
        return None

    @property
    def corrected(self) -> bool:
        final = self.final
        return final is not None and final.strip() != (self.machine or "").strip()


def read_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else sorted(base.rglob("*.jsonl"))
        for path in files:
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except ValueError:
                                continue
            except OSError:
                continue
    return rows


def segments(rows: list[dict[str, Any]]) -> list[Segment]:
    traces = {r["trace_id"]: r for r in rows if r.get("type") == "trace" and r.get("trace_id")}
    out: dict[tuple[str, str], Segment] = {}

    def seg(row) -> Segment:
        key = (row.get("trace_id") or "", row.get("segment_id") or "")
        if key not in out:
            trace = traces.get(key[0], {})
            out[key] = Segment(trace_id=key[0], segment_id=key[1],
                               target_language=trace.get("target_language", ""),
                               engine=trace.get("engine", ""), session=row.get("session"))
        return out[key]

    for row in rows:
        kind = row.get("type")
        if kind in ("score", "judgement") and row.get("target_language"):
            seg(row).target_language = seg(row).target_language or row["target_language"]
        if kind == "generation":
            s = seg(row)
            s.source, s.machine = row.get("source", ""), row.get("output", "")
            s.engine = row.get("engine") or s.engine
        elif kind == "score":
            s = seg(row)
            s.source = s.source or row.get("source", "")
            if not s.machine:
                s.machine = row.get("before", "")
            s.edits.append((float(row.get("at") or 0), row.get("after", "")))
        elif kind == "judgement":
            s = seg(row)
            s.source = s.source or row.get("source", "")
            s.machine = s.machine or row.get("output", "")
            s.approved = bool(row.get("approved"))
    return [s for s in out.values() if s.source]


def _messages(s: Segment) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"Translate the user's text to {s.target_language or 'the target language'}. Reply with the translation only."},
        {"role": "user", "content": s.source},
    ]


def to_sft(segs: list[Segment]) -> list[dict[str, Any]]:
    return [{"messages": _messages(s) + [{"role": "assistant", "content": s.final}],
             "meta": {"engine": s.engine, "corrected": s.corrected, "trace_id": s.trace_id}}
            for s in segs if s.final]


def to_dpo(segs: list[Segment]) -> list[dict[str, Any]]:
    return [{"prompt": _messages(s), "chosen": s.final, "rejected": s.machine,
             "meta": {"engine": s.engine, "trace_id": s.trace_id}}
            for s in segs if s.corrected and s.machine]


def to_tmx(segs: list[Segment], source_language: str = "en") -> str:
    units = []
    for s in segs:
        if not s.final:
            continue
        target = (s.target_language or "xx").replace(" ", "_")
        units.append(
            f'    <tu><tuv xml:lang="{escape(source_language)}"><seg>{escape(s.source)}</seg></tuv>'
            f'<tuv xml:lang="{escape(target)}"><seg>{escape(s.final)}</seg></tuv></tu>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<tmx version="1.4">\n'
            f'  <header creationtool="passage" srclang="{escape(source_language)}" '
            'datatype="plaintext" segtype="sentence" adminlang="en" o-tmf="passage"/>\n'
            '  <body>\n' + "\n".join(units) + '\n  </body>\n</tmx>\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m passage.export", description=__doc__.split("\n\n")[0])
    parser.add_argument("--format", choices=["sft", "dpo", "tmx"], required=True)
    parser.add_argument("--in", dest="inputs", action="append", default=[],
                        help="directory or file of trace JSONL (repeatable; default data/traces)")
    parser.add_argument("--out", help="output file (default stdout)")
    args = parser.parse_args(argv)

    inputs = args.inputs or [str(Path(__file__).resolve().parent.parent / "data" / "traces")]
    segs = segments(read_rows(inputs))
    if args.format == "tmx":
        text = to_tmx(segs)
        count = text.count("<tu>")
    else:
        records = to_sft(segs) if args.format == "sft" else to_dpo(segs)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        count = len(records)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(f"{count} {args.format} record(s) from {len(segs)} segment(s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
