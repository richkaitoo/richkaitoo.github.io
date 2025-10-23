---
layout: single
title: "My First Machine Learning Project"
excerpt: "A walkthrough of my predictive model and how I improved accuracy using feature engineering."
date: 2025-01-01
author: Ernest Essel-Kaitoo
author_profile: true
read_time: true
comments: true
share: true
related: true
tags: 
  - Machine Learning
  - Data Science
  - Python
categories: 
  - Projects
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "/assets/images/ml-project-banner.jpg"
  caption: "Exploring prediction with Random Forest"
class: wide
---



``` r
## if you haven't already installed jobbR
# devtools::install_github("dashee87/jobbR")

## loading the packages we'll need
require(jobbR)
require(ggplot2)

# collecting data scientist jobs in London from the Indeed API
dataScientists <- jobSearch(publisher = "yourpublisherID", query = "data+scientist",
country = "uk", location = "london", all = TRUE)

# collecting data analyst jobs in London from the Indeed API
dataAnalysts <- jobSearch(publisher = "yourpublisherID", query = "data+analyst",
country = "uk", location = "london", all = TRUE)
```
