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
  {% assign postlist = site.pages | where: "series", "kups-md-tutorials" | sort: "series_order" %}
  {% assign tutorial_count = postlist | size %}

  <h1>kUPS MD Tutorials</h1>
  <p class="blog-index-note">
    Executable molecular-dynamics notes for ML researchers who already know MLIPs and the equations of motion, but want the practical details behind initialization, integrators, ensembles, observables, free energies, enhanced sampling, and reliability checks.
  </p>
  <div class="blog-type-summary" aria-label="kUPS tutorial types">
    <span>Post types</span>
    <span>Tutorials {{ tutorial_count }}</span>
    <span>Executable notes</span>
    <span>MD practice</span>
    <span>MLIP reliability</span>
  </div>

  <ol class="bibliography">
  {% for post in postlist %}
    {% assign post_type = "tutorial" %}
    {% assign post_type_label = "Tutorial" %}
    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% if post.authors %}
      {% assign post_author_text = post.authors | join: ", " %}
    {% else %}
      {% assign post_author_text = post.author %}
    {% endif %}
    {% if post.last_updated %}
      {% assign post_date = post.last_updated %}
      {% assign date_label = "updated" %}
    {% else %}
      {% assign post_date = post.date %}
      {% assign date_label = "posted" %}
    {% endif %}
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
            <span class="blog-post-type blog-post-type-{{ post_type }}">{{ post_type_label }}</span>{% if post_author_text %}; {{ post_author_text }}{% endif %}; {{ date_label }} {{ post_date | date: '%B %d, %Y' }}; {{ read_time }} min read; part {{ post.series_order }} of {{ tutorial_count }}
          </div>
        </div>
      </div>
    </li>
  {% endfor %}
  </ol>
</div>
