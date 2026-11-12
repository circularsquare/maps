"""Fetch a URL to a file (certifi's CA bundle). python fetch.py <url> <out>"""
import sys, urllib.request, ssl, certifi
ctx = ssl.create_default_context(cafile=certifi.where())
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": "data-fetch/1.0"})
with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(sys.argv[2], "wb") as f:
    f.write(r.read())
print("ok")
