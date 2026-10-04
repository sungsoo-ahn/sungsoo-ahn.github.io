#!/usr/bin/env python3
"""Validate research-map annotations and generate reviewable analysis/keyword indexes.

Default invocation is read-only, including checks for generated report drift.
Use --write-reports after editing the canonical YAML inputs.
"""

import argparse
import collections
import datetime
import json
import math
import html
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORTS = {
    Path('_pages/research-map-analysis.md'): 'analysis',
    Path('docs/research-map-keywords.md'): 'keywords',
    Path('_includes/research_map_overview.liquid'): 'overview',
    Path('_includes/research_map_legend.liquid'): 'legend',
}
KINDS = ('domain', 'method', 'concept')
ROLES = {'central', 'used', 'evaluation', 'parallel'}
COLORS = {'purple', 'purple_strong', 'neutral', 'indigo', 'teal', 'amber', 'rose', 'green'}
SHAPES = {'circle', 'hexagon', 'square', 'cross', 'triangle', 'diagonal_cross', 'star', 'diamond'}
ID = re.compile(r'^[a-z][a-z0-9_]*$')
ARXIV = re.compile(r'^https://arxiv\.org/(?:html|pdf|abs)/(\d{4}\.\d{4,5})v[1-9]\d*(?:\.pdf)?$')


def load_inputs(root=ROOT):
    return tuple(yaml.safe_load((root / path).read_text()) for path in (
        '_data/research_map.yml', '_data/publications.yml', '_data/research_map_keywords.yml'))


def validate(data, publications, pool):
    """Return all actionable errors, rather than stopping at the first broken link."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def records(container, key):
        value = container.get(key) if isinstance(container, dict) else None
        require(isinstance(value, list) and bool(value), f'{key}: expected a nonempty list')
        if not isinstance(value, list):
            return []
        for i, item in enumerate(value):
            require(isinstance(item, dict), f'{key}[{i}]: expected an object')
        return [item for item in value if isinstance(item, dict)]

    def index(items, name):
        result = {}
        for item in items:
            key = item.get('id')
            if not isinstance(key, str) or not ID.fullmatch(key):
                errors.append(f'{name}: invalid ID {key!r}')
                continue
            require(key not in result, f'{name}: duplicate ID {key}')
            result[key] = item
        return result

    require(isinstance(data, dict) and data.get('schema_version') == 1, 'research_map: schema_version must be 1')
    require(isinstance(pool, dict) and pool.get('schema_version') == 1, 'keyword pool: schema_version must be 1')
    if not isinstance(data, dict) or not isinstance(pool, dict):
        return errors
    try:
        datetime.date.fromisoformat(str(data.get('reviewed_on')))
    except ValueError:
        errors.append('reviewed_on: expected an ISO date')
    themes = index(records(data, 'taxonomy'), 'taxonomy')
    works = index(records(data, 'works'), 'works')
    relationships = index(records(data, 'relationships'), 'relationships')
    keywords = index(records(pool, 'keywords'), 'keywords')
    contributions = index(records(data, 'contribution_categories'), 'contribution_categories')
    pubs = {p['id']: p for p in publications}

    for key, category in contributions.items():
        require(text(category.get('label')), f'{key}: contribution label required')
        require(isinstance(category.get('shape'), str) and category['shape'] in SHAPES, f'{key}: unsupported contribution shape')
    require(len({category.get('shape') for category in contributions.values() if isinstance(category.get('shape'), str)}) == len(contributions),
            'contribution_categories: shapes must be distinct')

    for key, theme in themes.items():
        require(theme.get('kind') in KINDS, f'{key}: invalid theme kind')
        require(theme.get('map_role') in {'major', 'detail'}, f'{key}: map_role must be major or detail')
        require(theme.get('color') in COLORS, f'{key}: color must reference the site palette')
        if 'display_color' in theme:
            require(isinstance(theme['display_color'], str) and re.fullmatch(r'#[0-9A-Fa-f]{6}', theme['display_color']),
                    f'{key}: display_color must be a six-digit hex color')
        if theme.get('kind') == 'concept':
            require(text(theme.get('map_label')), f'{key}: missing map_label')
        for field in ('label', 'description'):
            require(text(theme.get(field)), f'{key}: missing {field}')
    for key, item in list(themes.items()) + list(works.items()):
        for axis, limit in (('x', 1080), ('y', 720)):
            if axis not in item:
                continue  # Legacy coordinates are optional and unused by the timeline.
            value = item.get(axis)
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 <= value <= limit, f'{key}: invalid {axis} coordinate')

    for key, keyword in keywords.items():
        require(text(keyword.get('label')), f'{key}: keyword label missing')
        aliases = keyword.get('aliases')
        require(isinstance(aliases, list) and all(text(a) for a in aliases), f'{key}: aliases must be strings')
        if 'theme' in keyword:
            require(keyword['theme'] in themes, f'{key}: unknown keyword theme {keyword["theme"]}')

    assigned = collections.Counter()
    used_themes = collections.Counter()
    used_keywords = collections.Counter()
    for key, work in works.items():
        chronology = work.get('chronology', {})
        require(isinstance(chronology, dict), f'{key}: chronology must be an object')
        if not isinstance(chronology, dict):
            chronology = {}
        date = chronology.get('date')
        valid_date = isinstance(date, str) and bool(re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?', date))
        if valid_date:
            try:
                datetime.date.fromisoformat(date if len(date) == 10 else date + '-01')
            except ValueError:
                valid_date = False
        require(valid_date, f'{key}: chronology needs a valid ISO date or month')
        basis = chronology.get('basis')
        require(basis in {'conference_notification', 'arxiv_version', 'journal_month'}, f'{key}: invalid chronology basis')
        require(not valid_date or len(date) == (7 if basis == 'journal_month' else 10), f'{key}: chronology precision must match its basis')
        source = chronology.get('source')
        require(isinstance(source, str) and source.startswith('https://') and 'openreview.net' not in source,
                f'{key}: chronology needs an HTTPS source outside OpenReview')
        events = chronology.get('events')
        require(isinstance(events, list) and bool(events), f'{key}: chronology needs candidate events')
        verified_events = []
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict):
                errors.append(f'{key}: chronology event must be an object')
                continue
            event_date, event_basis, event_source = (event.get(field) for field in ('date', 'basis', 'source'))
            try:
                datetime.date.fromisoformat(event_date + '-01' if len(event_date) == 7 else event_date)
                require(len(event_date) == (7 if event_basis == 'journal_month' else 10),
                        f'{key}: event precision must match its basis')
                verified_events.append(event)
            except (TypeError, ValueError):
                errors.append(f'{key}: event needs a valid ISO date or month')
            require(event_basis in {'conference_notification', 'arxiv_version', 'journal_month'},
                    f'{key}: invalid event basis')
            require(isinstance(event_source, str) and event_source.startswith('https://') and 'openreview.net' not in event_source,
                    f'{key}: event needs an HTTPS source outside OpenReview')
            if event_basis == 'arxiv_version':
                version = event.get('version')
                require(isinstance(version, str) and bool(re.fullmatch(r'\d{4}\.\d{4,5}v[1-9]\d*', version)),
                        f'{key}: arXiv chronology must pin the eligible author version')
                require(event.get('author') == 'Sungsoo Ahn', f'{key}: arXiv chronology must verify authorship')
                require(event_source == f'https://arxiv.org/abs/{version}', f'{key}: arXiv event source must match its version')
        if verified_events:
            earliest = min(verified_events, key=lambda event: (event['date'] + '-01' if len(event['date']) == 7 else event['date'], event['basis']))
            require(all(chronology.get(field) == earliest.get(field) for field in ('date', 'venue', 'basis', 'source', 'version', 'author')),
                    f'{key}: chronology must select the earliest eligible event')
        excluded = chronology.get('excluded_arxiv_versions', [])
        require(isinstance(excluded, list), f'{key}: excluded_arxiv_versions must be a list')
        for event in excluded if isinstance(excluded, list) else []:
            require(isinstance(event, dict) and text(event.get('reason')) and text(event.get('source')),
                    f'{key}: excluded version needs its source and reason')
        for field in ('label', 'map_label', 'summary', 'contribution', 'limitation', 'evaluation'):
            require(text(work.get(field)), f'{key}: missing {field}')
        require(isinstance(work.get('map_contribution'), str) and work['map_contribution'] in contributions,
                f'{key}: needs one recognized map_contribution')
        require(text(work.get('map_contribution_reason')), f'{key}: map_contribution_reason required')
        secondary = work.get('map_secondary_contributions', [])
        require(isinstance(secondary, list), f'{key}: map_secondary_contributions must be a list')
        if isinstance(secondary, list):
            valid = [category for category in secondary if isinstance(category, str) and category in contributions]
            require(len(valid) == len(secondary), f'{key}: secondary contributions must use recognized categories')
            require(len(set(valid)) == len(valid), f'{key}: secondary contributions must be unique')
            require(work.get('map_contribution') not in valid, f'{key}: secondary contributions must exclude the main contribution')
        if 'map_layout' in work:
            hint = work['map_layout']
            require(isinstance(hint, dict), f'{key}: map_layout must be an object')
            if isinstance(hint, dict):
                require(not (hint.keys() - {'y', 'label_side', 'label_gap'}), f'{key}: unknown map_layout field')
                y = hint.get('y')
                require(isinstance(y, (int, float)) and not isinstance(y, bool)
                        and math.isfinite(y) and 8 <= y <= 292, f'{key}: map_layout y must be finite and within 8–292')
                if 'label_side' in hint:
                    require(hint['label_side'] in ('above', 'below'), f'{key}: map_layout label_side must be above or below')
                if 'label_gap' in hint:
                    gap = hint['label_gap']
                    require(isinstance(gap, (int, float)) and not isinstance(gap, bool)
                            and math.isfinite(gap) and 0 <= gap <= 24, f'{key}: map_layout label_gap must be finite and within 0–24')
        if 'overview' in work:
            require(isinstance(work['overview'], bool), f'{key}: legacy overview must be boolean')
        status = work.get('review_status')
        require(status in {'fulltext', 'partial'}, f'{key}: invalid review_status')
        if status == 'partial':
            require(text(work.get('coverage_note')), f'{key}: partial review needs a coverage_note')
        ids = work.get('publication_ids')
        require(isinstance(ids, list) and bool(ids), f'{key}: publication_ids required')
        ids = ids if isinstance(ids, list) else []
        for pub_id in ids:
            require(pub_id in pubs, f'{key}: unknown publication {pub_id}')
            assigned[pub_id] += 1
        if len(ids) > 1 and all(pub_id in pubs for pub_id in ids):
            same = {pubs[pub_id].get('arxiv') for pub_id in ids}
            require(len(same) == 1 and None not in same, f'{key}: grouped editions must share an arXiv identifier')
        sources = work.get('sources')
        require(isinstance(sources, list) and bool(sources), f'{key}: sources required')
        sources = sources if isinstance(sources, list) else []
        arxiv_ids = set()
        for source in sources:
            if not isinstance(source, dict):
                errors.append(f'{key}: source must be an object')
                continue
            raw_url = source.get('url', '')
            url = raw_url if isinstance(raw_url, str) else ''
            require(urlparse(url).scheme == 'https' and bool(urlparse(url).netloc), f'{key}: source URL must be HTTPS')
            require('openreview.net' not in url, f'{key}: prefer arXiv or an accessible author/publisher source')
            for field in ('label', 'locator'):
                require(text(source.get(field)), f'{key}: source {field} missing')
            if url.startswith('https://arxiv.org/'):
                match = ARXIV.fullmatch(url)
                require(bool(match), f'{key}: reviewed arXiv source must pin a version')
                if match:
                    arxiv_ids.add(match[1])
        for pub_id in ids:
            if pub_id in pubs and pubs[pub_id].get('arxiv'):
                require(re.sub(r'v\d+$', '', pubs[pub_id]['arxiv']) in arxiv_ids, f'{key}: source must use the arXiv link in publications.yml')
        memberships = work.get('memberships')
        require(isinstance(memberships, list) and bool(memberships), f'{key}: memberships required')
        memberships = memberships if isinstance(memberships, list) else []
        seen = set()
        kinds = set()
        for member in memberships:
            if not isinstance(member, dict):
                errors.append(f'{key}: membership must be an object')
                continue
            theme = member.get('theme')
            require(theme in themes, f'{key}: unknown theme {theme}')
            require(theme not in seen, f'{key}: duplicate membership {theme}')
            seen.add(theme)
            used_themes[theme] += 1
            if theme in themes:
                kinds.add(themes[theme]['kind'])
            require(member.get('role') in ROLES, f'{key}: invalid membership role')
            source = member.get('source')
            require(type(source) is int and 0 <= source < len(sources), f'{key}: membership source is out of range')
            require(text(member.get('locator')), f'{key}: membership locator required')
        require({'domain', 'method'} <= kinds, f'{key}: needs both domain and methodology memberships')
        for kind in ('domain', 'method'):
            require(any(isinstance(member, dict) and member.get('theme') in themes
                        and themes[member['theme']].get('kind') == kind
                        and themes[member['theme']].get('map_role') == 'major'
                        and member.get('role') != 'parallel' for member in memberships),
                    f'{key}: needs at least one non-parallel major {kind} theme')
        keys = work.get('keywords')
        require(isinstance(keys, list) and bool(keys), f'{key}: overcomplete keywords required')
        if isinstance(keys, list):
            require(len(keys) == len(set(keys)), f'{key}: duplicate keyword assignments')
            for keyword in keys:
                require(keyword in keywords, f'{key}: unknown keyword {keyword}')
                used_keywords[keyword] += 1
    for pub_id in pubs:
        require(assigned[pub_id] == 1, f'{pub_id}: expected one station, found {assigned[pub_id]}')
    for theme in themes:
        require(used_themes[theme] > 0, f'{theme}: unused theme (keep inactive terms in the keyword pool)')
    # Unassigned keywords are allowed: the pool is intentionally overcomplete and editable.
    pairs = set()
    for key, relation in relationships.items():
        require(text(relation.get('map_label')), f'{key}: missing map_label')
        a, b = relation.get('from'), relation.get('to')
        require(a in works and b in works and a != b, f'{key}: endpoints must be two different known works')
        pair = tuple(sorted((str(a), str(b))))
        require(pair not in pairs, f'{key}: duplicate endpoint pair; combine its explanation')
        pairs.add(pair)
        require(relation.get('status') in {'documented', 'interpretive'}, f'{key}: invalid connection status')
        for field in ('label', 'explanation', 'map_explanation'):
            require(text(relation.get(field)), f'{key}: missing {field}')
        for field in ('directed',):
            require(isinstance(relation.get(field), bool), f'{key}: {field} must be boolean')
        evidence = relation.get('evidence')
        require(isinstance(evidence, list), f'{key}: evidence must be a list')
        endpoints = set()
        for reference in evidence if isinstance(evidence, list) else []:
            if not isinstance(reference, dict):
                errors.append(f'{key}: evidence reference must be an object')
                continue
            endpoint = reference.get('work')
            require(endpoint in {a, b}, f'{key}: evidence must cite an endpoint')
            endpoints.add(endpoint)
            source = reference.get('source')
            source_count = len(works.get(endpoint, {}).get('sources', []))
            require(type(source) is int and 0 <= source < source_count, f'{key}: evidence source is out of range')
            require(text(reference.get('locator')), f'{key}: evidence locator missing')
        require(endpoints == {a, b}, f'{key}: connection needs evidence from both endpoints')
    for finding in records(data, 'findings'):
        require(text(finding.get('title')) and text(finding.get('text')), 'finding: title and text required')
        require(isinstance(finding.get('works'), list) and all(w in works for w in finding['works']), 'finding: unknown work')
    return errors


def page_header(title, permalink):
    return f'---\nlayout: page\ntitle: {json.dumps(title)}\npermalink: {permalink}\nnav: false\n---\n\n<!-- Generated by scripts/research_map.py. Edit the canonical YAML inputs. -->\n\n'


def timeline_layout(data, publications):
    """Use the canonical JS geometry for both the linked fallback and browser."""
    result = subprocess.run(
        ['node', str(ROOT / 'scripts/research_map_layout.mjs'), '--json'],
        input=json.dumps({'data': data, 'publications': publications}),
        text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def color_swatches(hex_color):
    """Keep route colors visible on both site backgrounds (3:1 contrast)."""
    base = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]

    def luminance(color):
        values = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        values = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * weight for v, weight in zip(values, (0.2126, 0.7152, 0.0722)))

    def adjust(background, target):
        result = hex_color
        for step in range(1, 21):
            a, b = luminance(result), luminance(background)
            if (max(a, b) + 0.05) / (min(a, b) + 0.05) >= 3.1:
                break
            result = '#' + ''.join(f'{math.floor(v + (target - v) * step / 20 + 0.5):02x}' for v in base)
        return result

    return adjust('#ffffff', 0), adjust('#21192b', 255)


def symbol_markup(layout, category, css_class='rm-symbol'):
    symbol = layout['symbols'][category['id']]
    attributes = {**symbol['attributes'], 'class': css_class, 'data-shape': category['shape']}
    attrs = ' '.join(f'{key}="{html.escape(str(value), quote=True)}"' for key, value in attributes.items())
    return f'<{symbol["tag"]} {attrs}/>'


def render_overview(data, publications):
    """Linked fallback with the browser's shared stations and routed paths."""
    palette = yaml.safe_load((ROOT / '_data/palette.yml').read_text())
    colors = {key: value['hex'] for family in ('brand', 'semantic', 'structure') for key, value in palette[family].items()}
    categories = {category['id']: category for category in data['contribution_categories']}
    layout = timeline_layout(data, publications)
    width, height = layout['width'], layout['height']

    def style(theme):
        light, dark = color_swatches(theme.get('display_color', colors[theme['color']]))
        return f'--rm-color-light:{light};--rm-color-dark:{dark}'

    result = ['<!-- Generated by scripts/research_map.py. Edit canonical research-map data. -->',
              '<div class="rm-timeline-scroll" tabindex="0" role="region" aria-label="Research papers by domain and contribution in publication order">',
              f'<div class="rm-timeline" style="--rm-track-width:{width}px">',
              f'<svg class="rm-network" width="{width}" height="{height}">']
    for row in layout['rows']:
        theme = row['theme']
        result.append(f'<g class="rm-route rm-themed" data-theme="{theme["id"]}" style="{style(theme)}">')
        for connection in row['connections']:
            title = html.escape(theme['label'])
            path = connection['path']
            result.append(f'<g class="rm-connection" data-from="{connection["from"]}" data-to="{connection["to"]}" data-connection="{connection["id"]}"><path d="{path}" class="rm-connection-casing" aria-hidden="true"/><path d="{path}" class="rm-connection-path"><title>{title}</title></path></g>')
        result.append('</g>')
    for station in layout['stations']:
        work, theme, x, y = station['work'], station['theme'], station['x'], station['y']
        editions = work['publications']
        category = categories[work['map_contribution']]
        domains = ' · '.join(item['label'] for item in station['themes'])
        families = ' · '.join(categories[family]['label'] for family in [work['map_contribution'], *work.get('map_secondary_contributions', [])])
        title = html.escape(work['title'] + ' · ' + work['summary'] + ' · ' + ', '.join(work['authors']) + ' · ' + ' · '.join(f'{edition["venue"]} {edition["year"]}' for edition in editions) + ' · ' + domains + ' · ' + families, quote=True)
        themes = ','.join(item['id'] for item in station['themes'])
        destination = html.escape(layout['paper_urls'][work['id']], quote=True)
        result.append(f'<a href="{destination}" target="_blank" rel="external nofollow noopener" class="rm-static-paper rm-themed" style="{style(theme)}" data-work="{work["id"]}" data-contribution="{category["id"]}" data-primary-label="true" data-instance="{station["id"]}" data-themes="{themes}" data-x="{x}" data-y="{y}" data-label-side="{station["labelSide"]}" aria-label="{title}" transform="translate({x},{y})"><circle class="rm-hit" r="16"/><circle class="rm-station-backplate" r="11" aria-hidden="true"/>' + symbol_markup(layout, category, 'rm-station') + f'<title>{title}</title><text class="rm-work-label" text-anchor="middle">')
        for line, baseline in zip(station['labelLines'], station['labelBaselines']):
            result.append(f'<tspan x="0" y="{baseline}">{html.escape(line)}</tspan>')
        result.append(f'</text><text class="rm-work-meta" text-anchor="middle" y="{station["metaY"]}">{html.escape(station["metaText"])}</text></a>')
    return '\n'.join(result + ['</svg></div></div>']) + '\n'


def render_legend(data, publications):
    """Stable domain-color and contribution-shape keys, sorted by first appearance."""
    palette = yaml.safe_load((ROOT / '_data/palette.yml').read_text())
    colors = {key: value['hex'] for family in ('brand', 'semantic', 'structure') for key, value in palette[family].items()}
    layout = timeline_layout(data, publications)
    rows = sorted(layout['rows'], key=lambda row: layout['x_by_id'][row['stations'][0]['id']])
    categories = {category['id']: category for category in data['contribution_categories']}
    result = ['<!-- Generated by scripts/research_map.py. Edit canonical research-map colors and contributions. -->',
              '<div class="rm-legend-group rm-domain-legend" aria-label="Domains"><span class="rm-legend-label">Domains</span>']
    for row in rows:
        theme = row['theme']
        light, dark = color_swatches(theme.get('display_color', colors[theme['color']]))
        result.append(f'<span class="rm-legend-item rm-themed" data-theme="{theme["id"]}" style="--rm-color-light:{light};--rm-color-dark:{dark}" title="{html.escape(theme["label"], quote=True)}"><i></i><span>{html.escape(theme.get("map_label", theme["label"]))}</span></span>')
    result.extend(['</div>', '<div class="rm-legend-group rm-contribution-legend" aria-label="Contributions"><span class="rm-legend-label">Contributions</span>'])
    for id in layout['contribution_order']:
        category = categories[id]
        result.append(f'<span class="rm-legend-item" data-contribution="{id}"><svg class="rm-contribution-symbol" viewBox="-12 -12 24 24" aria-hidden="true">' + symbol_markup(layout, category) + f'</svg><span>{html.escape(category["label"])}</span></span>')
    result.append('</div>')
    return '\n'.join(result) + '\n'



def render_reports(data, publications, pool):
    pubs = {p['id']: p for p in publications}
    works = {w['id']: w for w in data['works']}
    themes = {t['id']: t for t in data['taxonomy']}
    contributions = {c['id']: c for c in data['contribution_categories']}
    keywords = {k['id']: k for k in pool['keywords']}
    roles = {'central': 'central', 'used': 'used', 'evaluation': 'evaluation', 'parallel': 'conceptual parallel'}

    def work_link(id, full=False):
        w = works[id]
        label = pubs[w['publication_ids'][0]]['title'] if full else w['label']
        return f'[{label}](/research-map/analysis/#{id})'

    def source_link(work, reference):
        source = work['sources'][reference['source']]
        return f'[{work["label"]}: {reference["locator"]}]({source["url"]})'

    grouped = sum(len(w['publication_ids']) - 1 for w in works.values())
    fulltext = sum(w['review_status'] == 'fulltext' for w in works.values())
    counts = collections.Counter(r['status'] for r in data['relationships'])
    report = [page_header('Research map: paper analysis', '/research-map/analysis/'),
              f'Reviewed {data["reviewed_on"]}. The map covers **{len(publications)} publications**, grouped into **{len(works)} research works**, with **{len(data["relationships"])} curated connections** ({counts["documented"]} documented shared mechanisms and {counts["interpretive"]} conceptual parallels). {fulltext} works have full-text review; {len(works) - fulltext} remain partial. {grouped} edition pairs share stations.',
              '\n[Explore the homepage map](/#research-map) · [Overcomplete keyword index](/research-map/keywords/) · [Editing guide](/research-map/editing/)',
              '\n## Evidence and scope\n']
    report.extend('- ' + note for note in data['coverage_notes'])
    report.extend(['\nThe overview uses one shared station per work in a left-to-right chronological sweep. Papers are ordered by the earlier of their first arXiv version listing Sungsoo Ahn and their conference acceptance announcement. Journal publication months are a fallback when neither date is available. Labels and station symbols reserve clear space, and quiet periods contribute to variable spacing. This is not a proportional calendar scale. Blossom-BP (2015) therefore appears before GEGL (2020), regardless of category. Domain lines can bend through available space. A constrained force pass separates crowded stations and lines while preserving horizontal positions, clear labels, and routing corridors. Vertical movement is limited to 28 pixels. Routing favors horizontal runs and 45-degree transitions, and penalizes bends, detours, crossings, and close parallel tracks. Separate entry and exit positions keep domain tracks apart at shared stations. Background casings make clear gaps at unavoidable crossings, without line jumps. Busy interchanges receive extra horizontal room. Optional map_layout display hints refine crowded station and label positions after the force pass; local horizontal reflow preserves label clearance and chronology. The 300-pixel-high figure keeps the same geometry throughout navigation. Categories appear in a consistent legend below it, with distinct colors sorted by their first publication.',
                   '\nGraph and language modeling are represented as methodologies rather than overview domains. Their application papers are assigned to molecules, proteins, materials, or cells; generic neural graph learning, reasoning, control, and optimization are grouped under General ML: deep learning. General ML: graphical models covers the classical BP, partition-function, gauge, and elimination work. Rechecking graph experiments added molecular evaluation memberships for EPIC (Section 4.1 and Table 2) and Wavelet diffusion (QM9, Sections 4.1/4.4), and biological evaluation memberships for EPIC, Non-backtracking GNN (Peptides-func/struct), and Node diffusion (PPI). Every assignment retains its source locator.',
                   '\nThe map opens at the newest end. Left/right controls pan up to 360 CSS pixels at one fixed scale, retaining overlapping windows on narrow screens so no paper falls between steps; horizontal scrolling and Home/End are supported. Each paper has one nickname and first venue/year label above or below its station, shown when the label fits the viewport. Full titles, one-sentence summaries, author lists, all domains, main and secondary contributions, and editions appear on hover. Papers with multiple domain memberships act as interchanges where their colored routes meet. Major lines connect only adjacent publications within each complete theme sequence and never bridge an intermediate paper.',
                   '\nPaper stations link directly to arXiv in a new tab. Bibliographic arXiv links take precedence; annotated arXiv sources cover papers absent from the bibliography. Existing publisher, conference, or reviewed project sources provide destinations when no arXiv record is available. The homepage has no paper-wise conceptual view; curated connections and their evidence remain in this analysis and the editable annotations.',
                   '\nStation colors indicate domains and SVG shapes indicate one main contribution. The contribution families are Symmetry, Generative modeling, Representations, Learning & inference, Search & optimization, Reasoning, Agents, and Benchmarks. Their stable legends follow first appearance. Contribution annotations identify the central novelty rather than every method used. QHFlow and QHFlow2 share a Symmetry mark and an Electronic structure domain, alongside GPWNO. Secondary contribution labels remain in hover cards and never add station markers. MaskGXT / HACO uses an Agents marker and also lists Generative modeling. Other method annotations remain available on hover.',
                   '\nChronology selects the earlier of the first arXiv version listing Sungsoo Ahn and the official conference acceptance notification. Candidate dates and excluded pre-authorship versions are recorded below; journal months are a fallback when neither date is available. Venue/year labels remain bibliographic. Titles and identifiers break ties on the same date. Curated pairwise connections require evidence from both endpoints. The keyword pool is intentionally overcomplete and editable; keyword overlap never creates a scientific connection automatically.',
                   '\n## Cross-paper insights\n'])
    for finding in data['findings']:
        report.extend([f'### {finding["title"]}\n', finding['text'], '\nRelated works: ' + ', '.join(work_link(w) for w in finding['works']) + '.\n'])
    report.append('## Paper analyses\n')
    for work in works.values():
        id = work['id']
        publication = pubs[work['publication_ids'][0]]
        report.extend([f'<a id="{id}"></a>\n', f'### {work["label"]}\n',
                       f'**{publication["title"]}**\n',
                       ' · '.join(f'{pubs[p]["venue"]} {pubs[p]["year"]}' for p in work['publication_ids']) + f' · Review: {work["review_status"]}.\n',
                       work['summary'], '\n**Contribution.** ' + work['contribution'],
                       '\n**Map contribution.** ' + contributions[work['map_contribution']]['label'] + ': ' + work['map_contribution_reason'],
                       '\n**Evaluation.** ' + work['evaluation'], '\n**Evidence boundary.** ' + work['limitation']])
        if work.get('map_secondary_contributions'):
            report.append('\n**Other contributions.** ' + '; '.join(contributions[category]['label'] for category in work['map_secondary_contributions']) + '.')
        if work.get('coverage_note'):
            report.append('\n' + work['coverage_note'])
        chronology = work['chronology']
        report.append(f'\n**Chronology.** [{chronology["date"]}: {chronology["basis"].replace("_", " ")}]({chronology["source"]}).'
                      + (' ' + chronology['note'] if chronology.get('note') else ''))
        report.append('\n**Candidate dates.** ' + '; '.join(
            f'[{event["date"]}: {event["venue"]} {event.get("version", event["basis"].replace("_", " "))}]({event["source"]})'
            for event in chronology['events']) + '.')
        if chronology.get('excluded_arxiv_versions'):
            report.append('\n**Excluded versions.** ' + '; '.join(
                f'[{event["version"]} ({event["date"]})]({event["source"]}): {event["reason"]}'
                for event in chronology['excluded_arxiv_versions']))
        for kind, title in zip(KINDS, ('Domains', 'Methodologies', 'Concepts')):
            members = [m for m in work['memberships'] if themes[m['theme']]['kind'] == kind]
            if members:
                report.append(f'\n**{title}.** ' + '; '.join(f'{themes[m["theme"]]["label"]} ({roles[m["role"]]})' for m in members) + '.')
        report.append('\n**Keywords.** ' + '; '.join(keywords[k]['label'] for k in work['keywords']) + '.')
        report.append('\n**Sources.** ' + '; '.join(f'[{s["label"]}: {s["locator"]}]({s["url"]})' for s in work['sources']) + '.')
        relations = [r for r in data['relationships'] if id in (r['from'], r['to'])]
        if relations:
            report.append('\n**Specific connections.**\n')
            for r in relations:
                peer = r['to'] if r['from'] == id else r['from']
                report.append(f'- {work_link(peer)} — {r["label"]} ({r["status"]}); see [{r["id"]}](#{r["id"]}).')
        report.append('')
    report.append('\n## Curated connection evidence\n')
    for r in data['relationships']:
        report.extend([f'<a id="{r["id"]}"></a>\n', f'### {r["label"]}\n',
                       work_link(r['from']) + ' ↔ ' + work_link(r['to']) + f' · **{r["status"]}**.\n',
                       r['explanation'], '\n**Short explanation.** ' + r['map_explanation'],
                       '\nEvidence: ' + '; '.join(source_link(works[e['work']], e) for e in r['evidence']) + '.\n'])
    report.append('## Taxonomy definitions\n')
    for kind, title in zip(KINDS, ('Domains', 'Methodologies', 'Concepts')):
        report.append(f'### {title}\n')
        for theme in data['taxonomy']:
            if theme['kind'] == kind:
                report.append(f'- **{theme["label"]}** (`{theme["id"]}`; {theme["map_role"]}): {theme["description"]}')
        report.append('')

    keyword_report = [page_header('Research map: overcomplete keywords', '/research-map/keywords/'),
                      f'**{len(keywords)} editable keywords** across {len(works)} works. The pool includes broad themes, specific mechanisms, datasets, representations, and useful search aliases. It deliberately exceeds the plotted taxonomy.',
                      '\nEdit `_data/research_map_keywords.yml` to rename keywords or add aliases. Edit each work’s `keywords` in `_data/research_map.yml` to change assignments. Keywords and aliases are retained for editing and analysis; editing `memberships` changes the map’s categories. No connection is inferred automatically from this list.',
                      '\n[Editing guide](/research-map/editing/) · [Complete analysis](/research-map/analysis/) · [Homepage map](/#research-map)',
                      '\n## Keyword index\n']
    for keyword in sorted(keywords.values(), key=lambda k: k['label'].casefold()):
        members = [w['id'] for w in works.values() if keyword['id'] in w['keywords']]
        keyword_report.append(f'### {keyword["label"]}\n')
        keyword_report.append(f'ID: `{keyword["id"]}`' + (f' · Map category: `{keyword["theme"]}`' if keyword.get('theme') else '') + '.\n')
        if keyword['aliases']:
            keyword_report.append('Search aliases: ' + '; '.join(keyword['aliases']) + '.\n')
        if keyword.get('note'):
            keyword_report.append(keyword['note'] + '\n')
        keyword_report.append(', '.join(work_link(w) for w in members) + '.\n' if members else 'Unassigned; retained for future editing.\n')
    keyword_report.append('## Keywords by paper\n')
    for w in works.values():
        keyword_report.extend([f'### {w["label"]}\n', work_link(w['id'], True) + f' · Work ID: `{w["id"]}`.\n',
                               '; '.join(f'{keywords[k]["label"]} (`{k}`)' for k in w['keywords']) + '.\n'])
    return {'analysis': '\n'.join(report).rstrip() + '\n', 'keywords': '\n'.join(keyword_report).rstrip() + '\n',
            'overview': render_overview(data, publications), 'legend': render_legend(data, publications)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-reports', action='store_true', help='Regenerate the analysis and keyword index after validation')
    args = parser.parse_args(argv)
    data, publications, pool = load_inputs()
    errors = validate(data, publications, pool)
    if not errors:
        rendered = render_reports(data, publications, pool)
        for path, name in REPORTS.items():
            target = ROOT / path
            if args.write_reports:
                target.write_text(rendered[name])
            elif not target.exists() or target.read_text() != rendered[name]:
                errors.append(f'{path}: report drift; run python scripts/research_map.py --write-reports')
    if errors:
        print('\n'.join('ERROR: ' + error for error in errors))
        return 1
    print(f'Research map valid: {len(publications)} publications, {len(data["works"])} works, '
          f'{len(data["relationships"])} connections, {len(pool["keywords"])} keywords.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
