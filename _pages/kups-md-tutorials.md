---
layout: page
permalink: /kups-md-tutorials/
title: kUPS MD Tutorials
description: Executable molecular-dynamics tutorials for MLIP-aware machine-learning researchers.
nav: false
nav_order: 4
pagination:
  enabled: false
---

<div class="publications blog-index">
  {% assign tutorials = site.pages | where: "series", "kups-md-tutorials" | sort: "series_order" %}
  {% assign tutorial_count = tutorials | size %}

  <h1>kUPS Molecular Dynamics Tutorials</h1>
  <p class="blog-index-note">
    A hidden draft index for executable molecular-dynamics practice: initialization, integrators, ensembles, uncertainty, observables, free energies, enhanced sampling, and MLIP reliability.
  </p>
  <div class="blog-type-summary" aria-label="kUPS tutorial status">
    <span>Post types</span>
    <span>Tutorials {{ tutorial_count }}</span>
    <span>Draft series</span>
    <span>Hidden</span>
  </div>

  <p class="blog-index-note">
    These pages are intentionally kept out of public navigation while the simulations, figures, citations, and review notes mature. The source of truth for executable artifacts is <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.
  </p>

  <ol class="bibliography">
  {% for post in tutorials %}
    {% assign post_type = "tutorial" %}
    {% assign post_type_label = "Tutorial" %}
    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    <li>
      <div class="row">
        <div class="col-sm-10">
          <div class="title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </div>
          {% if post.description %}
            <div class="blog-list-description">{{ post.description }}</div>
          {% endif %}
          <div class="author">
            <span class="blog-post-type blog-post-type-{{ post_type }}">{{ post_type_label }} {{ post.series_order }}</span>{% if post.authors %}; {{ post.authors | join: ", " }}{% endif %}; {{ post.date | date: '%B %d, %Y' }}; {{ read_time }} min read
          </div>
        </div>
      </div>
    </li>
  {% endfor %}
  </ol>

  <h2>Repository</h2>
  <p>
    The repository owns configurations, notebooks, tests, compact summaries, manifests, figure generation, and review notes. Raw trajectories, model archives, and bulky caches are intentionally excluded.
  </p>

  <pre><code>git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run pytest</code></pre>
</div>
