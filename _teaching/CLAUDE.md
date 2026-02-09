# Teaching Notes Writing Guide

Instructions for writing and editing lecture notes in `_teaching/`.

**Skills:** `/academic-writing` for prose style, `/jekyll-writing` for rendering rules.

## Note Frontmatter

Every teaching note must include this frontmatter block:

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD
description: "One-sentence summary."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 2
preliminary: true
toc:
  sidebar: left
related_posts: false
---
```

## Author Note

Notes begin with an author note immediately after the frontmatter:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This is Preliminary Note N for the Protein &amp; Artificial Intelligence course...</em>
</p>
```

## Teaching-Specific Content Rules

- Biology students may not know ML terminology — always explain jargon on first use
- Build from concrete to abstract — introduce a concept with a specific example (e.g., one protein) before generalizing to the matrix/batch form
- Each section should have a clear prerequisite chain: what the student must already know, and what this section enables
- Start with a roadmap table mapping sections to "why it's needed"
- Footnotes work well for alternate terminology or historical context without cluttering the main text

## Figures

- Store teaching figure images in `assets/img/teaching/protein-ai/`
- Embed figures using raw HTML for layout control:

```html
<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/your_figure.png' | relative_url }}" alt="Description">
    <div class="caption mt-1">Caption text here.</div>
</div>
```
