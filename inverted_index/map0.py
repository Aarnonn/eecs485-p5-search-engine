#!/usr/bin/env python3
"""Map 0."""
import sys
import re


# Read from stdin
for line in sys.stdin:
    # Split the line into filename and html_content
    filename, html_content = line.split("\t", 1)
    
    # Removing non-alphanumeric characters & converting to lowercase
    text = re.sub(r"[^a-zA-Z0-9 ]+", "", html_content)
    text = text.casefold()
    
    # Splitting text into whitespace-delimited terms
    terms = text.split()
    
    # Remove stop words
    with open("stopwords.txt", encoding="utf-8") as stopwords_file:
        stopwords = set(line.strip() for line in stopwords_file)
    terms = [term for term in terms if term not in stopwords]
    
    # TODO: Output what you need for reduce0.py
    # For document count, you probably want to emit:
    # print(f"document_count\t1")
