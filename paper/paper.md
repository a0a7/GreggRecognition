---
title: 'Novel error-corrected deep learning approach to handwritten text recognition of Gregg shorthand'
tags:
  - Python
  - handwritten text recognition
  - pen stenography
  - shorthand
authors:
  - name: Alexander Weimer
    orcid: 0009-0008-5679-3042
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: University of Minnesota, United States
   index: 1
   ror: 017zqws13
date: 20 July 2025
bibliography: paper.bib

---

# Summary

Shorthand, also known as pen stenography, is a family of writing systems for English and
other languages that emerged out of a need for a fast and efficient writing system in a pre-digital age. Of the many English shorthand systems, Gregg shorthand is the most prevalent
[@zhai2018]. This paper presents GreggRecognition, a novel deep learning tool for handwritten text recognition of Gregg shorthand made for the purpose of preservation of shorthand documents with historical value. 

# Statement of need

Shorthand, also known as pen stenography, is a family of writing systems for English and other languages
that emerged out of a need for a fast and efficient
writing system in a pre-digital age.
Of the many English shorthand systems, Gregg
shorthand is the most
prevalent [@zhai2018]. With the advent of
digital text input and storage, shorthand has largely
fallen out of use in favor of standard typing and digital stenography [@rajasekaran2012].

The transliteration of short-hand scripts into regular text presents a unique challenge for several reasons. Shorthand characters
often lack distinct features, sometimes varying only
in length or degree of curvature [@Gregg2001Jan]. Furthermore, shorthand lexicons are often simplified, often missing vowels or other defining features of words. While the human mind can accommodate for these kinds of omissions, creating a digital system that can do the same poses a challenge.

Reading of manuscripts written in shorthand is
vital to understanding documents in a wide variety
of fields where time-efficient handwriting was necessary in the past, such as law and medicine. The
digitization and thereby preservation of shorthand
documents, therefore, presents possible benefits in
preservation of history and culture. Further, the development of extensible HTR and OCR systems, in
this case with a focus on English shorthand, feasibly
opens avenues for the creation of HTR and OCR
systems for other written languages—thus presenting
possible benefits for the preservation of world languages and cultures.

GreggRecognition, a novel deep learning approach to handwritten text recognition of Gregg shorthand was developed to meet this need. GreggRecognition is based on the Gated Recurrent Unit (GRU) architecture, and was developed using the Pytorch framework. It was trained on the Gregg-1916 dataset introduced by Zhai et. al. in 2018, comprising greater than 16,000 Gregg shorthand words [@zhai2018].

# Acknowledgements

We acknowledge non-financial support from Patricia Price, Gus Davidson, Zackary Pace and Henry Doten during the development of this project as part of the Minnetonka Research program.

# References
