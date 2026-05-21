---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: false
---

<div class="post">

<header class="blog-header">
  <h1>Technical Notes</h1>
  <p class="blog-subtitle">Intuitive explanations of physics, chemistry, and generative modeling concepts for ML researchers.</p>
  <p class="blog-disclaimer">
    These posts are written with AI assistance through interactive editing and direction by the author, as an ongoing experiment in AI-assisted technical writing and figure generation.
  </p>
</header>

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% comment %} Header removed — blog_name and blog_description hidden {% endcomment %}

{% comment %} Tag/category list removed {% endcomment %}

{% assign featured_posts = site.posts | where: "featured", "true" %}
{% if featured_posts.size > 0 %}
<br>

<div class="container featured-posts">
{% assign is_even = featured_posts.size | modulo: 2 %}
<div class="row row-cols-{% if featured_posts.size <= 2 or is_even == 0 %}2{% else %}3{% endif %}">
{% for post in featured_posts %}
<div class="col mb-4">
<a href="{{ post.url | relative_url }}">
<div class="card hoverable">
<div class="row g-0">
<div class="col-md-12">
<div class="card-body">
<div class="float-right">
<i class="fa-solid fa-thumbtack fa-xs"></i>
</div>
<h3 class="card-title text-lowercase">{{ post.title }}</h3>
<p class="card-text">{{ post.description }}</p>

                    {% if post.external_source == blank %}
                      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
                    {% else %}
                      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
                    {% endif %}
                    {% assign year = post.date | date: "%Y" %}

                    <p class="post-meta">
                      {{ read_time }} min read &nbsp; &middot; &nbsp;
                      <a href="{{ year | prepend: '/blog/' | relative_url }}">
                        <i class="fa-solid fa-calendar fa-sm"></i> {{ year }} </a>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </a>
        </div>
      {% endfor %}
      </div>
    </div>
    <hr>

{% endif %}

  <div class="post-list blog-card-list">

    {% assign postlist = site.posts | sort: "date" | reverse %}

    {% for post in postlist %}

    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <article class="blog-card">
      {% if post.thumbnail %}
      <div class="blog-card-media">
        <img src="{{ post.thumbnail | relative_url }}" alt="">
      </div>
      {% endif %}
      <div class="blog-card-body">
        <h2 class="blog-card-title">
          {% if post.redirect == blank %}
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {% elsif post.redirect contains '://' %}
            <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }}</a>
          {% else %}
            <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
          {% endif %}
        </h2>
        <p class="blog-card-summary">{{ post.description }}</p>
        <div class="blog-card-meta">
          <span>{{ read_time }} min read</span>
          <span>{{ post.date | date: '%B %d, %Y' }}</span>
          {% if post.external_source %}
          <span>{{ post.external_source }}</span>
          {% endif %}
        </div>
        {% if tags != "" %}
        <div class="blog-card-tags">
          {% for tag in post.tags %}
            <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </article>

    {% endfor %}

  </div>


</div>
