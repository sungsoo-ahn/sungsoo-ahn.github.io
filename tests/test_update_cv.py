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
        self.assertEqual(cv_source.count(caio_title), 2)
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
        self.assertIn("+82 10-9495-1392", cv_source)

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

    def test_funding_matches_project_ledger(self):
        cv_source = Path("cv/cv.tex").read_text(encoding="utf-8")
        for grant_id in (
            "PJT-25-100009",
            "N10250153",
            "RS-2025-21063030",
            "RS-2025-02304967",
            "RS-2025-02216257",
            "RS-2024-00436165",
            "RS-2022-NR072184",
            "KSC-2025-CRE-0602",
            "RS-2025-02653113",
            "KSC-2024-CRE-0535",
        ):
            self.assertIn(grant_id, cv_source)

        grants = cv_source.split("\\section{Grants}", 1)[1]
        self.assertNotIn("{\\small", grants)
        research_grants, computing_grants = grants.split(
            "\\noindent\\textsc{Research Computing Awards}", 1
        )
        self.assertIn("\\noindent\\textsc{Research}", research_grants)
        self.assertNotIn(" million", research_grants)
        self.assertNotIn("PI;", grants)
        self.assertNotIn("Co-PI", grants)
        self.assertIn("Principal Investigator, K-MELLODDY", research_grants)
        self.assertIn("Project 3 Principal Investigator", research_grants)
        self.assertIn("KRW 11.5B total project funding", research_grants)
        self.assertIn("KRW 1.5B total project funding", research_grants)
        self.assertIn("KRW 2.0B total project funding", research_grants)
        self.assertIn("KRW 600M total direct costs", research_grants)
        self.assertNotIn("per year", research_grants)
        self.assertNotIn("allocation in", research_grants)
        self.assertIn(
            "\\grantentry{Integrated Representation Learning for Biomolecular Foundation Models}",
            computing_grants,
        )
        self.assertNotIn("K-Fold", computing_grants)
        self.assertNotIn("{KFold}", computing_grants)
        self.assertIn("256 NVIDIA B200 GPUs, Project PJT-25-100009", computing_grants)
        self.assertNotIn("KRW 17.4B", computing_grants)
        self.assertNotIn("Korea Electronics Technology Institute", computing_grants)
        self.assertNotIn("Brev", computing_grants)
        for allocation in (
            "8 NVIDIA H100 GPUs",
            "56 NVIDIA H200 GPUs",
            "32 NVIDIA B200 GPUs",
            "16,000 NVIDIA A100 GPU-hours",
            "36 Nurion KNL and 10 Neuron allocations",
        ):
            self.assertIn(allocation, computing_grants)
        self.assertNotIn("GPU server", computing_grants)
        self.assertNotIn("eight-GPU server", computing_grants)
        self.assertNotIn("for six months", computing_grants)
        self.assertNotIn("for one year", computing_grants)
        for project_title in (
            "Towards a Unified Generative Model for Molecular Interactions in Materials",
            "Development of a Materials Generative Foundation Model with Molecular-Materials Interactions",
            "Generative Materials Foundation Models for Adsorbate Complex Design",
            "Generative Models for Designing Metal-Organic Frameworks for Direct Air Capture",
            "Developing Universal Coarse-Grained Atomistic Foundation Models",
            "Chemical Language and Logic for Artificial Chemical Intelligence",
        ):
            self.assertIn(f"\\grantentry{{{project_title}}}", computing_grants)
        self.assertIn("32 NVIDIA H100 GPUs, Project RS-2025-02653113", computing_grants)
        self.assertNotIn(";", cv_source)
        self.assertNotIn("\\datedline", grants)
        self.assertEqual(grants.count("\\grantentry{"), 16)
        self.assertNotIn("\\textperiodcentered", cv_source)
        self.assertIn("\\textit{#1}, #3, #2. #4.", cv_source)
        self.assertNotIn("Research Grants (continued)", grants)
        self.assertNotIn("Korea Advanced Institute of Science and Technology", cv_source)
        self.assertNotIn("Pohang University of Science and Technology", cv_source)

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
        self.assertEqual((len(conferences), len(journals), len(preprints)), (58, 6, 10))
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
