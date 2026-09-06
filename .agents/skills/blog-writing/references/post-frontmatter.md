# Post metadata and opening note

Requirements are owned by [_posts/AGENTS.md](../../../../_posts/AGENTS.md);
the validator is [validate_blog.py](../../../../scripts/validate_blog.py).
Use real values from the article. This template intentionally includes only
the ordinary fields; the bracketed values are placeholders.

```yaml
---
layout: post
title: "[Article title]"
date: YYYY-MM-DD
last_updated: YYYY-MM-DD
description: "[One-sentence description]"
post_type: tutorial
editorial_status: ai-generated
authors: ["Sungsoo Ahn"]
categories: [category-slug]
tags: [topic]
toc:
  sidebar: left
related_posts: false
---
```

Use a category from [blog_categories.yml](../../../../_data/blog_categories.yml).
Choose `post_type` from `tutorial`, `technical-note`, or `research`.
Author names must reflect the actual authors.

Optional fields:

- `abstract`: a short abstract displayed below the metadata.
- `blog_blocks`: a list of `title` and Markdown `content` pairs for useful
  named opening blocks.
- `selected: true`: only for explicitly curated Selected posts.
- `series`, `series_title`, `series_description`, `series_order`: retain when
  the article belongs to an intentional series.
- `order` is legacy; the blog index sorts by publication date.

Do not add every optional field to a new post. Editorial status records review
provenance; neither publication date nor a humanizing edit changes it.

Immediately after frontmatter, use the existing author-note shape, with brief
factual context relevant to that article:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: [Reader-facing context or acknowledgment.]</em>
</p>
```
