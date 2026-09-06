---
name: site-validation
description: Validate changes to this Jekyll site or open its local preview. Use for metadata/assets, generated-content drift, instruction hygiene, rendering, and visual checks; does not regenerate content merely to check it.
---

# Validation and preview

Choose checks from the changed behavior. Complete required checks once and
revisit them when new changes or failures justify it. Report a missing tool or
environment limitation explicitly.

## Check selection

| Change | Check |
| --- | --- |
| Instructions or skills | `python3 scripts/validate_agent_hygiene.py` |
| Blog content, metadata, assets, lecture manifests | `python3 scripts/validate_blog.py` |
| Publication data | `uv run python scripts/update_publications.py --check` |
| CV sources or generation | `uv run python scripts/update_cv.py --check --no-compile` and relevant tests |
| Python logic | `uv run python -m unittest discover -s tests -p 'test_<module>.py'` |
| Hidden kUPS pages | `python3 scripts/validate_kups_pages.py` |
| Jekyll config, layouts, includes, CSS, or substantial rendered content | Build and inspect affected pages |

The member importer writes its output; it is not a check. Prefer the existing
validators over duplicating their schemas in prose.

Some existing tests invoke generators (including the CV tests). Inspect their
side effects before running them for a read-only request; use isolated fixtures
or the explicit check commands when regeneration is not authorized.

For render-sensitive content, read
[Jekyll rendering](references/jekyll-rendering.md).
For layout, typography, or substantial visual changes, read
[visual review](references/visual-review.md).

## Build without disturbing the preview

On this Mac, use Homebrew Ruby. Put an isolated validation build in a fresh
temporary directory:

```bash
site_build_dir=$(mktemp -d)
PATH="/opt/homebrew/opt/ruby/bin:$PATH" /opt/homebrew/opt/ruby/bin/bundle exec jekyll build --destination "$site_build_dir"
```

Use the actual returned directory when inspecting artifacts. A build does not
require killing the user's server or changing content to clear unrelated drift.

## Open a preview when requested

1. Check `lsof -iTCP:4000 -sTCP:LISTEN -n -P`.
2. If Jekyll is already serving, reuse it. If another process owns the port,
   identify it without terminating it.
3. If the port is free, start and leave running:

```bash
PATH="/opt/homebrew/opt/ruby/bin:$PATH" /opt/homebrew/opt/ruby/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

4. Open `http://127.0.0.1:4000/blog/` for a blog preview, or the requested
   page's route. Do not restart a running server unless explicitly requested.

For visual validation, inspect desktop and a narrow viewport together, fix
observed defects, and confirm the affected state. Additional passes need a
specific unresolved concern. Do not claim visual verification from a build alone.
