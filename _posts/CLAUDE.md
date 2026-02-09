# Blog Writing Guide

Instructions for writing and editing blog posts in `_posts/`.

**Skills:** `/blog-writing` for prose style, `/jekyll-writing` for rendering rules.

## Post Frontmatter

Every post must include this frontmatter block:

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD
last_updated: YYYY-MM-DD
description: "One-sentence summary for SEO and post listings."
order: 1  # controls display order (lower = appears later in list)
categories: [category-name]
tags: [tag1, tag2, tag3]
toc:
  sidebar: left
related_posts: false
---
```

**MANDATORY: Update `last_updated`** — Whenever you edit a blog post, update the `last_updated` field in the frontmatter to today's date. This is required for every edit, no matter how small.

## Author Note

Posts begin with an author note immediately after the frontmatter:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: Your context here — background, acknowledgments, recommended further reading.</em>
</p>
```

## Blog-Specific Content Rules

- Collapse related works into paragraphs rather than giving each its own subsection
- Readers likely don't know domain-specific jargon (e.g., "quadrupole") — use plain language
- Footnotes work well for alternate terminology without cluttering the main text
- When referencing publications, add to a References section as you cite, not at the end

## Figures

- Store figure images in `assets/img/blog/`
- Write a single Python script with separate functions per figure — easier to regenerate individually
- Use a consistent color palette across all figures with matching text colors for equation-figure correspondence
- Embed figures using the Liquid include:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_figure.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Your caption here." %}
```

## CSS

- Distinct h2/h3 styling (border-bottom on h2, different font sizes) for visual hierarchy
- These are one-time additions that carry over to future posts
