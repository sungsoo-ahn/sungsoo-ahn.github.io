import copy
import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('research_map', ROOT / 'scripts/research_map.py')
map_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(map_module)


class ResearchMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.publications, cls.pool = map_module.load_inputs()

    def errors_for(self, mutate):
        data = copy.deepcopy(self.data)
        mutate(data)
        return map_module.validate(data, self.publications, self.pool)

    def test_corpus_and_reports_are_consistent(self):
        self.assertEqual(map_module.validate(self.data, self.publications, self.pool), [])
        reports = map_module.render_reports(self.data, self.publications, self.pool)
        for path, name in map_module.REPORTS.items():
            self.assertEqual((ROOT / path).read_text(), reports[name])

    def test_new_publication_requires_an_annotation(self):
        publication = {**self.publications[0], 'id': 'newpaper'}
        errors = map_module.validate(self.data, self.publications + [publication], self.pool)
        self.assertTrue(any('newpaper: expected one station' in e for e in errors))

    def test_duplicate_station_assignment_is_rejected(self):
        errors = self.errors_for(lambda d: d['works'][1]['publication_ids'].append(d['works'][0]['publication_ids'][0]))
        self.assertTrue(any('found 2' in e for e in errors))

    def test_connection_requires_both_endpoint_sources(self):
        errors = self.errors_for(lambda d: d['relationships'][0]['evidence'].pop())
        self.assertTrue(any('needs evidence from both endpoints' in e for e in errors))

    def test_invalid_evidence_index_is_rejected(self):
        errors = self.errors_for(lambda d: d['relationships'][0]['evidence'][0].update(source=10))
        self.assertTrue(any('evidence source is out of range' in e for e in errors))

    def test_keywords_and_theme_assignments_cannot_drift(self):
        errors = self.errors_for(lambda d: d['works'][0]['keywords'].append('missing_keyword'))
        self.assertTrue(any('unknown keyword missing_keyword' in e for e in errors))
        errors = self.errors_for(lambda d: d['works'][0]['memberships'][0].update(theme='missing_theme'))
        self.assertTrue(any('unknown theme missing_theme' in e for e in errors))

    def test_arxiv_review_pins_a_version(self):
        errors = self.errors_for(lambda d: d['works'][0]['sources'][0].update(url='https://arxiv.org/abs/2602.07351'))
        self.assertTrue(any('must pin a version' in e for e in errors))

    def test_malformed_nested_evidence_has_an_actionable_error(self):
        errors = self.errors_for(lambda d: d['relationships'][0]['evidence'].append(None))
        self.assertTrue(any('evidence reference must be an object' in e for e in errors))
        errors = self.errors_for(lambda d: d['works'][0]['sources'][0].update(url=None))
        self.assertTrue(any('source URL must be HTTPS' in e for e in errors))

    def test_partial_review_stays_visible(self):
        def mutate(d):
            next(w for w in d['works'] if w['review_status'] == 'partial').pop('coverage_note')
        self.assertTrue(any('partial review needs' in e for e in self.errors_for(mutate)))

    def test_unassigned_keywords_are_allowed_in_overcomplete_pool(self):
        pool = copy.deepcopy(self.pool)
        pool['keywords'].append({'id': 'future_topic', 'label': 'Future topic', 'aliases': []})
        self.assertEqual(map_module.validate(self.data, self.publications, pool), [])

    def test_every_paper_requires_permanent_major_theme_coverage(self):
        errors = self.errors_for(lambda data: next(theme for theme in data['taxonomy'] if theme['id'] == 'd_materials').update(map_role='detail'))
        self.assertTrue(any('major domain theme' in error for error in errors))

    def test_chronology_requires_verified_source_and_valid_precision(self):
        errors = self.errors_for(lambda data: data['works'][0]['chronology'].update(date='2026-13-05'))
        self.assertTrue(any('valid ISO date or month' in error for error in errors))
        errors = self.errors_for(lambda data: data['works'][0]['chronology'].update(source='https://openreview.net/example'))
        self.assertTrue(any('outside OpenReview' in error for error in errors))
        errors = self.errors_for(lambda data: data['works'][0]['chronology'].update(date='2026-12'))
        self.assertTrue(any('precision must match' in error for error in errors))

    def test_chronology_rejects_later_events_and_unverified_authorship(self):
        errors = self.errors_for(lambda data: data['works'][0]['chronology'].update(date='2026-12-06'))
        self.assertTrue(any('earliest eligible event' in error for error in errors))
        errors = self.errors_for(lambda data: data['works'][0]['chronology']['events'][0].pop('author'))
        self.assertTrue(any('verify authorship' in error for error in errors))
        errors = self.errors_for(lambda data: data['works'][0]['chronology']['events'][0].update(source='https://arxiv.org/abs/2602.07351'))
        self.assertTrue(any('source must match its version' in error for error in errors))

    def test_fallback_includes_all_papers_and_only_adjacent_segments(self):
        overview = ET.fromstring(map_module.render_overview(self.data, self.publications))
        layout = map_module.timeline_layout(self.data, self.publications)
        stations = [node for node in overview.iter('a') if node.get('class') == 'rm-static-paper rm-themed']
        self.assertEqual(len(stations), 74)
        for station in stations:
            work = next(work for work in self.data['works'] if work['id'] == station.get('data-work'))
            self.assertEqual(station.get('href'), layout['paper_urls'][work['id']])
            self.assertEqual(station.get('target'), '_blank')
            self.assertIn(work['summary'], station.find('title').text)
            self.assertIn('noopener', station.get('rel'))
        mask = next(station for station in stations if station.get('data-work') == 'seong2026discovering')
        self.assertIn('Agents', mask.find('title').text)
        self.assertIn('Generative modeling', mask.find('title').text)

        self.assertEqual({node.get('data-work') for node in stations}, {work['id'] for work in self.data['works']})
        self.assertTrue(all(float(node.get('data-x')) == layout['positions'][node.get('data-instance')] for node in stations))
        self.assertTrue(all(float(node.get('data-y')) == next(station['y'] for station in layout['stations'] if station['id'] == node.get('data-instance')) for node in stations))
        self.assertEqual(len([node for node in overview.iter('svg') if node.get('class') == 'rm-network']), 1)
        self.assertEqual(layout['height'], 300)
        for theme_row in layout['rows']:
            theme = theme_row['theme']['id']
            route = next(node for node in overview.iter('g') if node.get('data-theme') == theme)
            members = theme_row['stations']
            edges = [node for node in route.iter('g') if node.get('class') == 'rm-connection']
            self.assertEqual([(node.get('data-from'), node.get('data-to')) for node in edges],
                             [(a['id'], b['id']) for a, b in zip(members, members[1:])])
        self.assertEqual(len([node for node in overview.iter('text') if node.get('class') == 'rm-work-label']), 74)
        self.assertTrue(all(node.get('r') == '9' for node in overview.iter('circle') if node.get('class') == 'rm-station'))
        self.assertTrue(all(node.get('r') == '16' for node in overview.iter('circle') if node.get('class') == 'rm-hit'))
        self.assertNotIn('rm-timeline-axis', map_module.render_overview(self.data, self.publications))
        self.assertNotIn('rm-row-heading', map_module.render_overview(self.data, self.publications))
        self.assertNotIn('rm-identity-path', map_module.render_overview(self.data, self.publications))
        self.assertTrue(all('A' not in node.get('d', '') for node in overview.iter('path') if node.get('class') == 'rm-connection-path'))
        legend = ET.fromstring('<div>' + map_module.render_legend(self.data, self.publications) + '</div>')
        self.assertEqual(len([node for node in legend.iter('span') if node.get('data-theme')]), 8)
        self.assertEqual(len([node for node in legend.iter('span') if node.get('data-contribution')]), 8)
        genetic = next(work for work in self.data['works'] if work['id'] == 'ahn2020guiding')
        self.assertLess(layout['x_by_id']['ahn2015minimum'], layout['x_by_id'][genetic['id']])

    def test_display_names_are_required_and_do_not_replace_scientific_names(self):
        errors = self.errors_for(lambda data: data['works'][0].pop('map_label'))
        self.assertTrue(any('missing map_label' in error for error in errors))
        errors = self.errors_for(lambda data: data['relationships'][0].pop('map_label'))
        self.assertTrue(any('missing map_label' in error for error in errors))
        self.assertEqual(len(self.pool['keywords']), 412)

    def test_contribution_annotations_require_one_known_category_and_rationale(self):
        errors = self.errors_for(lambda d: d['works'][0].pop('map_contribution'))
        self.assertTrue(any('recognized map_contribution' in error for error in errors))
        errors = self.errors_for(lambda d: d['works'][0].update(map_contribution=['generation', 'symmetry']))
        self.assertTrue(any('recognized map_contribution' in error for error in errors))
        errors = self.errors_for(lambda d: d['works'][0].pop('map_contribution_reason'))
        self.assertTrue(any('map_contribution_reason required' in error for error in errors))
        errors = self.errors_for(lambda d: d['contribution_categories'][0].update(shape='octagon'))
        self.assertTrue(any('unsupported contribution shape' in error for error in errors))

    def test_secondary_contributions_are_optional_known_unique_and_distinct(self):
        for secondary in [None, "generation", ["unknown"], [[]], ["search", "search"]]:
            with self.subTest(secondary=secondary):
                errors = self.errors_for(lambda d: d['works'][0].update(map_secondary_contributions=secondary))
                self.assertTrue(any('secondary' in error for error in errors))
        errors = self.errors_for(lambda d: d['works'][0].update(map_secondary_contributions=[d['works'][0]['map_contribution']]))
        self.assertTrue(any('exclude the main contribution' in error for error in errors))
        errors = self.errors_for(lambda d: d['works'][0].update(map_secondary_contributions=[]))
        self.assertEqual(errors, [])

    def test_display_hints_reject_invalid_geometry_and_unknown_fields(self):
        invalid = [None, {}, {'y': float('nan')}, {'y': float('inf')}, {'y': True},
                   {'y': 7}, {'y': 293}, {'y': 164, 'label_side': 'left'},
                   {'y': 164, 'label_side': []}, {'y': 164, 'label_gap': -1},
                   {'y': 164, 'label_gap': 25}, {'y': 164, 'label_gap': True},
                   {'y': 164, 'label_gap': float('nan')}, {'y': 164, 'x': 42}]
        for hint in invalid:
            with self.subTest(hint=hint):
                errors = self.errors_for(lambda data: data['works'][0].update(map_layout=hint))
                self.assertTrue(any('map_layout' in error for error in errors))
        errors = self.errors_for(lambda data: data['works'][0].update(map_layout={'y': 164}))
        self.assertEqual(errors, [])

    def test_legacy_geometry_and_overview_flags_are_optional(self):
        data = copy.deepcopy(self.data)
        for item in data['works'] + data['taxonomy']:
            for key in ('x', 'y', 'overview'):
                item.pop(key, None)
        self.assertEqual(map_module.validate(data, self.publications, self.pool), [])
        self.assertEqual(map_module.render_overview(data, self.publications),
                         map_module.render_overview(self.data, self.publications))


if __name__ == '__main__':
    unittest.main()
