---
layout: archive
title: "Blogs"
permalink: /blog/
author_profile: true
class: wide
---
{% assign written_year = '' %}
{% assign all_posts = site.posts | default: [] %}
{% for post in all_posts %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% if year != written_year %}
    {% assign written_year = year %}
  {% endif %}
  {% include archive-single.html %}
{% endfor %}
