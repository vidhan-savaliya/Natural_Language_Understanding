"""
Problem 1 Scraper for IIT Jodhpur

"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import io
import warnings
from collections import deque

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

try:
    from PyPDF2 import PdfReader
    PDF_OK = True
    print("PyPDF2 loaded - PDF extraction is go.")
except ImportError:
    PDF_OK = False
    print("Need PyPDF2! Run: pip install PyPDF2")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Safari/537.36"
}

PDF_DIR = "downloaded_pdfs"


def make_dirs():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)


def on_iitj_domain(url):
    return "iitj.ac.in" in urlparse(url).netloc


def looks_like_pdf(url):
    return url.lower().endswith(".pdf")


def is_junk_file(url):
    junk = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.zip', '.rar', '.tar', '.gz', '.mp3', '.mp4',
            '.avi', '.mov', '.xls', '.xlsx', '.ppt', '.pptx',
            '.doc', '.docx', '.css', '.js', '.woff', '.woff2',
            '.ttf', '.eot')
    return url.lower().endswith(junk)


def grab_pdf_text(url):
    """Downloads a PDF and rip out every single line of text"""
    if not PDF_OK:
        return ""

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20, verify=False)
        resp.raise_for_status()
    except Exception as e:
        print(f" PDF grab failed: {e}")
        return ""

    # Saves a local copy
    fname = url.split("/")[-1].split("?")[0]
    if not fname.endswith(".pdf"):
        fname = "document.pdf"

    save_to = os.path.join(PDF_DIR, fname)
    n = 1
    while os.path.exists(save_to):
        base, ext = os.path.splitext(fname)
        save_to = os.path.join(PDF_DIR, f"{base}_{n}{ext}")
        n += 1

    with open(save_to, "wb") as f:
        f.write(resp.content)

    # It reads every page, every line of website
    try:
        reader = PdfReader(io.BytesIO(resp.content))
        chunks = []
        for page_num, page in enumerate(reader.pages):
            raw = page.extract_text()
            if raw:
                # Filter PDF text: keep only lines that look like actual text and paragraphs
                lines = raw.split('\n')
                for line in lines:
                    line = line.strip()
                    # Keep lines longer than 60 characters with multiple words
                    if len(line) > 60 and line.count(' ') > 5:
                        chunks.append(line)

        full = "\n".join(chunks)
        print(f"   => Got {len(full):,} chars of high-quality text from {fname} ({len(reader.pages)} pages)")
        return full

    except Exception as e:
        print(f"   [X] PDF read error on {fname}: {e}")
        return ""


def scrape_one_page(url):
    """Hit a normal HTML page, grab ALL text and discover every link"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12, verify=False)
        resp.raise_for_status()
    except Exception as e:
        print(f"   [X] Skip: {e}")
        return "", []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Grab text ONLY from <p> tags and some main article tags, avoiding UI/nav noise
    all_tags = soup.find_all(["p", "article", "blockquote"])
    lines = []
    for tag in all_tags:
        txt = tag.get_text(strip=True)
        # ONLY keep large paragraphs
        if len(txt) > 100 and txt.count(' ') > 10:
            lines.append(txt)

    page_text = "\n".join(lines)

    # Now find every link on the page
    links_found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href).split("#")[0]

        if on_iitj_domain(full):
            links_found.append(full)

    return page_text, links_found


def mega_crawl(seeds, max_html_pages=500, max_pdfs=200):
    """
    Full website crawler.
    
    Crawling up to max_html_pages web pages
    Downloading up to max_pdfs PDF documents
    PDFs do NOT count against the page limit, they run separately
    """
    make_dirs()

    print()
    print("=" * 55)
    print("  IIT JODHPUR FULL DOMAIN CRAWLER")
    print("  Web page limit: " + str(max_html_pages))
    print("  PDF limit: " + str(max_pdfs))
    print("=" * 55)
    print()

    queue = deque(seeds)
    visited = set(seeds)

    web_texts = []
    pdf_texts = []

    html_count = 0
    pdf_count = 0

    while queue and (html_count < max_html_pages):
        url = queue.popleft()

        if looks_like_pdf(url):
            if pdf_count < max_pdfs:
                print(f"[PDF #{pdf_count+1}] {url}")
                txt = grab_pdf_text(url)
                if txt:
                    pdf_texts.append(txt)
                    pdf_count += 1
                time.sleep(0.3)
            continue

        if is_junk_file(url):
            continue

        print(f"[HTML {html_count+1}/{max_html_pages}] {url}")
        text, links = scrape_one_page(url)

        if text:
            web_texts.append(text)

        new = 0
        for link in links:
            if link not in visited:
                visited.add(link)
                queue.append(link)
                new += 1

        html_count += 1

        if html_count % 20 == 0:
            print(f"  Progress: {html_count} pages, {pdf_count} PDFs, {len(visited)} total URLs found")

        time.sleep(0.3)

    # After html crawl, mop up any remaining PDFs in the queue
    print("\nMopping up remaining PDF links from queue...")
    while queue and pdf_count < max_pdfs:
        url = queue.popleft()
        if looks_like_pdf(url) and pdf_count < max_pdfs:
            print(f"[PDF #{pdf_count+1}] {url}")
            txt = grab_pdf_text(url)
            if txt:
                pdf_texts.append(txt)
                pdf_count += 1
            time.sleep(0.3)

    # Build the monster corpus
    print()
    print("=" * 55)
    print("CRAWL RESULTS")
    print("=" * 55)
    print(f"HTML pages scraped  : {html_count}")
    print(f"PDFs downloaded     : {pdf_count}")
    print(f"Total URLs seen     : {len(visited)}")

    parts = []
    for i, t in enumerate(web_texts):
        parts.append(f"=== WEB DOCUMENT {i+1} ===\n{t}")

    for i, t in enumerate(pdf_texts):
        parts.append(f"=== PDF DOCUMENT {i+1} ===\n{t}")

    giant_corpus = "\n\n".join(parts)

    with open("additional_scraped_corpus.txt", "w", encoding="utf-8") as f:
        f.write(giant_corpus)

    print(f"Corpus size         : {len(giant_corpus):,} characters")
    print(f"Saved to additional_scraped_corpus.txt")
    print("DONE!")


if __name__ == "__main__":
    # Multiple entry points across ALL known IITJ subdomains
    entry_points = [
        "https://iitj.ac.in/",
        "https://iitj.ac.in/academics/",
        "https://iitj.ac.in/department/",
        "https://iitj.ac.in/uploaded_docs/",
        "https://cse.iitj.ac.in/",
        "https://ee.iitj.ac.in/",
        "https://ai.iitj.ac.in/",
        "https://me.iitj.ac.in/",
        "https://phy.iitj.ac.in/",
        "https://math.iitj.ac.in/",
        "https://hss.iitj.ac.in/",
        "https://civil.iitj.ac.in/",
        "https://che.iitj.ac.in/",
        "https://bio.iitj.ac.in/",
        "https://mems.iitj.ac.in/",
        "https://iitj.ac.in/research/",
        "https://iitj.ac.in/placement/",
    ]

    mega_crawl(seeds=entry_points, max_html_pages=100, max_pdfs=20)
