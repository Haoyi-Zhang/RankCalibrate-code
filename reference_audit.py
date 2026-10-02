#!/usr/bin/env python3
"""Offline, deterministic bibliography and source-ledger audit.

The audit deliberately does not attempt network access.  It verifies that the
curated primary-record ledger agrees exactly with the checked-in BibTeX and the
external-resource ledger.  Human/source verification scope is recorded rather
than inferred from URL reachability.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


def _strip_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '{' and value[-1] == '}') or
                            (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


def _split_top_level(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == '"' and depth == 0:
            quoted = not quoted
        elif not quoted:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth < 0:
                    raise ValueError('Unbalanced closing brace in BibTeX entry')
            elif char == ',' and depth == 0:
                pieces.append(text[start:index].strip())
                start = index + 1
    if depth != 0 or quoted:
        raise ValueError('Unbalanced braces or quotes in BibTeX entry')
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def parse_bibtex(path: Path | str) -> dict[str, BibEntry]:
    """Parse the checked-in BibTeX subset with balanced-brace handling."""
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    entries: dict[str, BibEntry] = {}
    index = 0
    while True:
        at = text.find('@', index)
        if at < 0:
            break
        brace = text.find('{', at + 1)
        if brace < 0:
            raise ValueError(f'Malformed BibTeX entry after byte {at}')
        entry_type = text[at + 1:brace].strip().lower()
        if not entry_type or not entry_type.replace('_', '').isalnum():
            raise ValueError(f'Invalid BibTeX entry type near byte {at}')
        depth = 1
        quoted = False
        escaped = False
        end = brace + 1
        while end < len(text) and depth:
            char = text[end]
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"' and depth == 1:
                quoted = not quoted
            elif not quoted:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
            end += 1
        if depth:
            raise ValueError(f'Unterminated BibTeX entry near byte {at}')
        body = text[brace + 1:end - 1].strip()
        comma = body.find(',')
        if comma < 1:
            raise ValueError(f'Missing BibTeX key near byte {at}')
        key = body[:comma].strip()
        if key in entries:
            raise ValueError(f'Duplicate BibTeX key: {key}')
        fields: dict[str, str] = {}
        for piece in _split_top_level(body[comma + 1:]):
            if not piece:
                continue
            if '=' not in piece:
                raise ValueError(f'Malformed field in {key}: {piece!r}')
            name, value = piece.split('=', 1)
            name = name.strip().lower()
            if not name or name in fields:
                raise ValueError(f'Duplicate or empty field {name!r} in {key}')
            fields[name] = _strip_value(value)
        entries[key] = BibEntry(entry_type, key, fields)
        index = end
    if not entries:
        raise ValueError(f'No BibTeX entries found in {path}')
    return entries


def _read_unique_rows(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key_field not in reader.fieldnames:
            raise ValueError(f'{path.name} is missing required key column {key_field!r}')
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, 2):
            key = (row.get(key_field) or '').strip()
            if not key:
                raise ValueError(f'{path.name}:{line_number}: empty {key_field}')
            if key in rows:
                raise ValueError(f'{path.name}:{line_number}: duplicate {key_field} {key!r}')
            rows[key] = {name: (value or '').strip() for name, value in row.items()}
    return rows


def audit_references(
    bib_path: Path | str,
    audit_path: Path | str,
    resources_path: Path | str,
    *,
    minimum_entries: int = 55,
) -> dict[str, int]:
    """Fail closed if bibliography, reference audit, or source ledger drift."""
    bib_path = Path(bib_path)
    audit_path = Path(audit_path)
    resources_path = Path(resources_path)
    entries = parse_bibtex(bib_path)
    if len(entries) < minimum_entries:
        raise ValueError(
            f'Bibliography has {len(entries)} entries; minimum is {minimum_entries}'
        )

    audit = _read_unique_rows(audit_path, 'bib_key')
    if set(audit) != set(entries):
        missing = sorted(set(entries) - set(audit))
        extra = sorted(set(audit) - set(entries))
        raise ValueError(f'Reference-audit key drift: missing={missing}, extra={extra}')

    required = {
        'bib_key', 'title', 'authors', 'year', 'venue_or_status',
        'persistent_id', 'primary_url', 'source_tier', 'metadata_check',
        'content_check', 'checked_on',
    }
    with audit_path.open(newline='', encoding='utf-8') as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    absent = sorted(required - fieldnames)
    if absent:
        raise ValueError(f'{audit_path.name} missing columns: {absent}')

    resource_rows: list[dict[str, str]] = []
    with resources_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        needed = {'scholarly_or_official_url', 'supported_claim'}
        if not reader.fieldnames or not needed <= set(reader.fieldnames):
            raise ValueError(f'{resources_path.name} lacks columns {sorted(needed)}')
        resource_rows = [
            {name: (value or '').strip() for name, value in row.items()}
            for row in reader
        ]
    resources: dict[str, str] = {}
    for row in resource_rows:
        claim = row['supported_claim']
        if claim in entries:
            if claim in resources:
                raise ValueError(f'Duplicate scholarly resource row for {claim}')
            resources[claim] = row['scholarly_or_official_url']
    if set(resources) != set(entries):
        missing = sorted(set(entries) - set(resources))
        extra = sorted(set(resources) - set(entries))
        raise ValueError(f'External-resource key drift: missing={missing}, extra={extra}')

    allowed_tiers = {'primary_publication', 'official_preprint'}
    allowed_metadata = {'primary_record_checked', 'publisher_record_checked'}
    allowed_content = {
        'full_text_or_claim_sections_checked',
        'abstract_and_relevant_passage_checked',
        'metadata_only_not_used_for_technical_claim',
    }
    identifiers: dict[str, str] = {}
    for key, entry in entries.items():
        row = audit[key]
        for field in required - {'persistent_id'}:
            if not row[field]:
                raise ValueError(f'{key}: empty audit field {field}')
        for field, bib_field in (('title', 'title'), ('authors', 'author'), ('year', 'year')):
            actual = entry.fields.get(bib_field, '')
            if row[field] != actual:
                raise ValueError(
                    f'{key}: audit {field} does not match BibTeX: '
                    f'{row[field]!r} != {actual!r}'
                )
        if row['primary_url'] != resources[key]:
            raise ValueError(f'{key}: primary URL differs from external_resources.csv')
        if not row['primary_url'].startswith('https://'):
            raise ValueError(f'{key}: primary URL must use https')
        if row['source_tier'] not in allowed_tiers:
            raise ValueError(f'{key}: invalid source tier {row["source_tier"]!r}')
        if row['metadata_check'] not in allowed_metadata:
            raise ValueError(f'{key}: invalid metadata check {row["metadata_check"]!r}')
        if row['content_check'] not in allowed_content:
            raise ValueError(f'{key}: invalid content check {row["content_check"]!r}')
        try:
            checked = date.fromisoformat(row['checked_on'])
        except ValueError as error:
            raise ValueError(f'{key}: checked_on is not ISO date') from error
        if checked > date.today():
            raise ValueError(f'{key}: checked_on lies in the future')
        identifier = row['persistent_id']
        if identifier:
            normalized = identifier.lower()
            if normalized in identifiers:
                raise ValueError(
                    f'Duplicate persistent ID {identifier!r}: '
                    f'{identifiers[normalized]} and {key}'
                )
            identifiers[normalized] = key

    return {
        'entries': len(entries),
        'primary_publications': sum(
            row['source_tier'] == 'primary_publication' for row in audit.values()
        ),
        'official_preprints': sum(
            row['source_tier'] == 'official_preprint' for row in audit.values()
        ),
        'full_text_or_claim_sections': sum(
            row['content_check'] == 'full_text_or_claim_sections_checked'
            for row in audit.values()
        ),
    }


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bib', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('resources', type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = audit_references(args.bib, args.audit, args.resources)
    print(
        'reference audit: '
        f'{summary["entries"]} entries; '
        f'{summary["primary_publications"]} primary publications, '
        f'{summary["official_preprints"]} official preprints; '
        f'{summary["full_text_or_claim_sections"]} full-text/claim-section checks.'
    )


if __name__ == '__main__':
    main()
