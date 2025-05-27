import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    pages = list(corpus.keys())
    all_pages_num = len(pages)
    probabilities = dict()
    
    
    links = corpus[page]
    number_of_page_links = len(links)
    
    base_probability = (1 - damping_factor) / all_pages_num
    
    for page in pages:
        probabilities[page] = base_probability
    
    if number_of_page_links > 0:
        link_probability = damping_factor/number_of_page_links
        
        for link in links:
            probabilities[link] += link_probability
    
    else:
        for page in pages:
            probabilities[page] = 1/all_pages_num
            
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    pages = list(corpus.keys())
    
    current_page = random.choice(pages)
    
    counts = {}
    
    for page in pages:
        counts[page] = 0
        
    counts[current_page] += 1
    
    for i in range(n-1):
        
        probabilities = transition_model(corpus, current_page,damping_factor)
        
        pages = list(probabilities.keys())
        weights = list(probabilities.values())
        
        next_page = random.choices(pages, weights=weights, k=1)[0]
        current_page = next_page
    
    pagerank = {}
    
    for page, count in counts.items():
        pagerank[page]=count/n
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {}
    
    for page in corpus:
        pagerank[page] = 1/N
        
    treshhold = 0.001
    
    while True:
        new_rank = {}
        
        for page in corpus:
            total = 0
            
            for other_page in corpus: 
                
                if page in corpus[other_page]:
                    total += pagerank[other_page] / len(corpus[other_page])
                    
                elif len(corpus[other_page]) == 0:
                    total += pagerank[other_page]/N
    
            new_rank[page] = (1 - damping_factor)/N + damping_factor * total
        
        small_diff = True
        for page in corpus:
            if abs(new_rank[page] - pagerank[page]) > treshhold:
                small_diff = False
                break
        pagerank = new_rank
        if small_diff:
            break

if __name__ == "__main__":
    main()
