---
layout: page
permalink: /kups-md-tutorials/
title: kUPS MD Tutorials
description: An executable introduction to molecular dynamics, from physical ideas and JAX algorithms to production kUPS simulations.
nav: false
nav_order: 4
pagination:
  enabled: false
---

<div class="publications blog-index">
  {% assign published_tutorials = site.posts | where: "series", "kups-md-tutorials" %}
  {% if published_tutorials.size > 0 %}
    {% assign kups_foundations = site.pages | where: "permalink", "/kups-md-tutorials/foundations/" %}
    {% assign postlist = kups_foundations | concat: published_tutorials | sort: "series_order" %}
  {% else %}
    {% assign postlist = site.pages | where: "series", "kups-md-tutorials" | sort: "series_order" %}
  {% endif %}
  {% assign tutorial_count = postlist | size %}

  <h1>kUPS MD Tutorials</h1>
  <p class="blog-index-note">
    An executable introduction for readers who know Python and elementary calculus but may be new to molecular dynamics. Each lesson connects physical intuition, equations, a compact JAX implementation, the corresponding kUPS interface, and an interpreted simulation result.
  </p>
  <div class="blog-type-summary" aria-label="kUPS tutorial types">
    <span>Reading path</span>
    <span>Lessons {{ tutorial_count }}</span>
  </div>

  <ol class="bibliography">
  {% for post in postlist %}
  {% assign post_type = post.post_type | default: "tutorial" %}
  {% assign post_type_label = post_type | replace: "-", " " | capitalize %}
  {% if post_type == "technical-note" %}
    {% assign post_type_label = "Technical note" %}
  {% endif %}
  {% if post.authors %}
    {% assign post_author_text = post.authors | join: ", " %}
  {% else %}
    {% assign post_author_text = post.author %}
  {% endif %}

  {% if post.external_source == blank %}
    {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
  {% else %}
    {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
  {% endif %}

    <li>
      <div class="row">
        <div class="col-sm-10">
          <div class="title">
            {% if post.redirect == blank %}
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {% elsif post.redirect contains '://' %}
              <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }}</a>
            {% else %}
              <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
            {% endif %}
          </div>
          {% if post.description %}
            <div class="blog-list-description">{{ post.description }}</div>
          {% endif %}
          <div class="author">
            <span class="blog-post-type blog-post-type-{{ post_type }}">{% if post.series_order == 0 %}Start here{% else %}{{ post_type_label }}{% endif %}</span>{% if post_author_text %}; {{ post_author_text }}{% endif %}; {{ post.date | date: '%B %d, %Y' }}; {{ read_time }} min read<span class="sr-only">; lesson {{ forloop.index }} of {{ tutorial_count }}</span>
          </div>
        </div>
      </div>
    </li>
  {% endfor %}
  </ol>
</div>
