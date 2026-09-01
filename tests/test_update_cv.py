import unittest
from pathlib import Path

import scripts.update_cv as update_cv


def generated_output() -> str:
    entries = update_cv.load_publications()
    conferences = [e for e in entries if e["type"] == "conference"]
    journals = [e for e in entries if e["type"] == "journal"]
    preprints = [
        e
        for e in entries
        if e["type"] == "preprint" and str(e.get("arxiv", "")).strip()
    ]
    conferences.sort(key=update_cv.sort_key)
    journals.sort(key=update_cv.sort_key)
    preprints.sort(key=update_cv.preprint_sort_key)
    return "\n".join(
        update_cv.format_entry(e, category)
        for category, group in (
            ("conference", conferences),
            ("journal", journals),
            ("preprint", preprints),
        )
        for e in group
    )


class UpdateCvTest(unittest.TestCase):
    def test_structured_sections_match_generated_cv_source(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        data = update_cv.load_cv_data()
        self.assertEqual(update_cv.expected_cv_source(cv_source, data), cv_source)
        for section in update_cv.render_cv_sections(data):
            self.assertEqual(cv_source.count(f"% SYNC:{section}:BEGIN"), 1)
            self.assertEqual(cv_source.count(f"% SYNC:{section}:END"), 1)

    def test_committed_pdf_matches_sources_and_website_copy(self):
        self.assertEqual(
            Path("cv/source.sha256").read_text(encoding="utf-8").strip(),
            update_cv.source_digest(),
        )
        self.assertEqual(Path("cv/cv.pdf").read_bytes(), Path("assets/pdf/cv.pdf").read_bytes())

    def test_plain_text_is_safely_escaped_for_latex(self):
        self.assertEqual(
            update_cv.tex_escape_plain("KAIST–Mila & 50%"),
            r"KAIST--Mila \& 50\%",
        )

    def test_talk_titles_and_dates_are_consistent(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        caio_title = "비정형 데이터 기반 딥러닝의 과학기술 활용 및 혁신 사례"
        genu_title = "From Atomistic Generative Models to Amortized Free-Energy Estimation"
        self.assertEqual(cv_source.count(caio_title), 3)
        self.assertIn("그래프신경망의 과학기술 활용 및 혁신 사례", cv_source)
        self.assertEqual(cv_source.count("KAIST Chief AI Officer Program, Seoul"), 4)
        self.assertNotIn("KAIST Chief AI Officer Program, Daejeon", cv_source)
        for new_title in (
            "Amortizing Molecular Simulation with Neural Networks",
            "Recent Trends in AI for Science",
        ):
            self.assertIn(new_title, cv_source)
        self.assertIn(
            f"\\cventry{{{genu_title}}}"
            "{Generative Models and Uncertainty Quantification (GenU), "
            "Copenhagen, Denmark}{Sep. 2026}",
            cv_source,
        )
        self.assertNotIn("(forthcoming)", cv_source)
        self.assertNotIn("\\hfill", cv_source)
        for inconsistent_name in (
            "Intl. Conference",
            "KAIST-MILA",
            "Metal Organic Frameworks",
            "Physics Informed Machine Learning",
        ):
            self.assertNotIn(inconsistent_name, cv_source)

    def test_small_cv_consistency_edits_are_preserved(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        self.assertIn("\\section{Invited Talks and Seminars}", cv_source)
        self.assertNotIn("\\section{Talks}", cv_source)
        self.assertIn("\\section{Teaching}", cv_source)
        self.assertNotIn("\\section{Courses}", cv_source)
        self.assertIn("pdftitle={Curriculum Vitae - Sungsoo Ahn}", cv_source)
        self.assertIn("pdfauthor={Sungsoo Ahn}", cv_source)
        self.assertIn("{Email:}", cv_source)
        self.assertNotIn("{Mobile:}", cv_source)
        self.assertNotIn("\\section{Grants}", cv_source)

        data = update_cv.load_cv_data()
        courses = data["courses"]
        intro_ai = [course for course in courses if course["title"].startswith("Introduction to Artificial Intelligence")]
        pgm = [course for course in courses if course["title"].startswith("Probabilistic Graphical Models")]
        self.assertEqual(len(intro_ai), 1)
        self.assertEqual(intro_ai[0]["date"], "Fall 2023 and Fall 2024")
        self.assertEqual(len(pgm), 1)
        self.assertEqual(pgm[0]["date"], "Spring 2022 and Spring 2023")

    def test_sections_use_single_line_entry_format(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        self.assertIn("\\newcommand{\\cventry}", cv_source)
        self.assertIn("\\newcommand{\\cvunnumberedentry}", cv_source)
        self.assertNotIn("\\datedline", cv_source)
        self.assertNotIn("tabularx", cv_source)
        self.assertNotIn("\\textit{KAIST}", cv_source)
        self.assertNotIn("\\textit{POSTECH}", cv_source)
        self.assertNotIn("\\newpage", cv_source)

    def test_public_cv_excludes_private_data(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        data = update_cv.load_cv_data()
        self.assertNotIn("grants", data)
        self.assertNotIn("mobile", data["contact"])
        self.assertEqual(update_cv.render_cv_sections(data)["GRANTS"], "")
        self.assertNotIn("\\section{Grants}", cv_source)
        self.assertNotIn("{Mobile:}", cv_source)

    def test_private_overlay_adds_private_sections_without_mutating_public_data(self):
        public_data = update_cv.load_cv_data()
        overlay = {
            "contact": {"mobile": "private-mobile"},
            "entry_overrides": {
                "service": {
                    "ai_cred_mentor": {"organization": "Private program reference"}
                }
            },
            "grants": {
                "research": [
                    {
                        "title": "Private research project",
                        "organization": "Principal Investigator, Private sponsor",
                        "date": "2026",
                        "detail": "Private amount and identifier",
                    }
                ],
                "computing": [],
            },
        }
        private_data = update_cv.apply_private_overlay(
            public_data, overlay, Path("private-overlay.yml")
        )
        private_sections = update_cv.render_cv_sections(private_data)
        self.assertEqual(private_data["contact"]["mobile"], "private-mobile")
        self.assertIn("Private program reference", private_sections["SERVICE"])
        self.assertIn("\\section{Grants}", private_sections["GRANTS"])
        self.assertIn("Private research project", private_sections["GRANTS"])
        self.assertNotIn("mobile", public_data["contact"])

    def test_publication_counts_and_icml_2026_acceptances(self):
        entries = update_cv.load_publications()
        conferences = [e for e in entries if e["type"] == "conference"]
        journals = [e for e in entries if e["type"] == "journal"]
        preprints = [
            e
            for e in entries
            if e["type"] == "preprint" and str(e.get("arxiv", "")).strip()
        ]
        icml_2026 = [
            e for e in conferences if update_cv.get_abbr(e) == "ICML" and e["year"] == 2026
        ]
        self.assertEqual((len(conferences), len(journals), len(preprints)), (58, 6, 11))
        self.assertEqual(len(icml_2026), 6)

    def test_conferences_use_reverse_chronological_order(self):
        entries = update_cv.load_publications()
        conferences_2024 = sorted(
            [
                e
                for e in entries
                if update_cv.get_year(e) == 2024
                and update_cv.get_abbr(e) in {"NeurIPS", "ICML", "ICLR"}
            ],
            key=update_cv.sort_key,
        )
        first_index = {
            abbr: next(i for i, entry in enumerate(conferences_2024) if update_cv.get_abbr(entry) == abbr)
            for abbr in ("NeurIPS", "ICML", "ICLR")
        }
        self.assertLess(first_index["NeurIPS"], first_index["ICML"])
        self.assertLess(first_index["ICML"], first_index["ICLR"])

    def test_rendering_exceptions_are_preserved(self):
        output = generated_output()
        self.assertIn("Martin Ester, Jinkyoo Park", output)
        self.assertNotIn("and et al.", output)
        self.assertIn("in \\textit{Findings of ACL}", output)
        self.assertIn("in \\textit{Findings of EMNLP}", output)
        self.assertIn("in \\textit{KDD, Datasets and Benchmarks Track}", output)
        for venue in ("ICML", "ICLR", "NeurIPS", "AISTATS", "IJCAI"):
            self.assertIn(f"in \\textit{{{venue}}}", output)
        for verbose_venue in (
            "International Conference on Machine Learning",
            "International Conference on Learning Representations",
            "Conference on Neural Information Processing Systems",
        ):
            self.assertNotIn(verbose_venue, output)
        self.assertIn("Graph Generation with $K^2$-trees", output)
        self.assertIn("2019(12), 124015", output)
        self.assertIn("64(3), 1471--1480", output)
        self.assertIn("oral presentation (86 of 7,304 submissions, 1.2\\%)", output)
        self.assertNotIn(", In \\textit", output)
        self.assertNotIn("accept rate", output)
        self.assertNotIn("first NeurIPS oral", output)
        self.assertIn("Sungsoo Ahn}†", output.replace("$^\\dagger$", "†"))
        self.assertIn("Insu Han†", output.replace("$^\\dagger$", "†"))
        self.assertIn("Shell Xu Hu", output)
        self.assertIn("Neil D. Lawrence", output)
        self.assertIn("Sung-Ju Hwang", output)
        self.assertIn("Rafael G\\'{o}mez-Bombarelli", output)

    def test_daggers_mark_equal_corresponding_authors(self):
        entries = update_cv.load_publications()
        daggered_entries = [entry for entry in entries if any("†" in author for author in entry["authors"])]
        self.assertGreater(len(daggered_entries), 0)
        for entry in daggered_entries:
            self.assertGreaterEqual(sum("†" in author for author in entry["authors"]), 2, entry["title"])

    def test_four_recent_preprints_are_present(self):
        output = generated_output()
        for title in (
            "Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist",
            "MADField: Multi-fidelity Amortized Density Field",
            "Atom-level Protein Representation Learning Improves Protein Structure Prediction",
            "VibeProteinBench: An Evaluation Benchmark",
        ):
            self.assertIn(title, output)
        self.assertIn(
            "A Systematic Evaluation of Co-folding Model Representations for Small-Molecule Learning",
            output,
        )
        ordered_titles = (
            "Progressive Multi-Agent Reasoning",
            "AtomMOF: All-Atom Flow Matching",
            "INDIBATOR: Diverse and Fact-Grounded Individuality",
            "A Systematic Evaluation of Co-folding Model Representations",
        )
        positions = [output.index(title) for title in ordered_titles]
        self.assertEqual(positions, sorted(positions))

    def test_publication_section_labels_and_links_are_consistent(self):
        update_cv.main(["--no-compile"])
        rendered = Path("cv/publications.tex").read_text(encoding="utf-8")
        for heading in ("Conference Papers", "Journal Articles", "Preprints"):
            self.assertIn(f"\\textsc{{{heading}}}", rendered)
        for old_heading in ("\\textsc{Conference}", "\\textsc{Journal}", "\\textsc{Preprint}"):
            self.assertNotIn(old_heading, rendered)
        self.assertIn("{[paper]}", rendered)
        self.assertNotIn("{[OpenReview]}", rendered)
        self.assertNotIn("{[PMLR]}", rendered)
        self.assertNotIn("{[NeurIPS]}", rendered)
        self.assertIn("{[arXiv]}", rendered)
        self.assertIn("{[code]}", rendered)
        self.assertIn("{[project]}", rendered)
        self.assertIn("{[dataset]}", rendered)

        latent_veracity = next(
            line for line in rendered.splitlines() if "Latent Veracity Inference" in line
        )
        for label in ("paper", "arXiv", "code"):
            self.assertIn(f"{{[{label}]}}", latent_veracity)


if __name__ == "__main__":
    unittest.main()
