import arxiv


def discover_papers(topic: str, max_results: int = 5):
    """
    Fetch papers relevant to `topic`, then present the most recent ones
    first. We still search by relevance (so we get topically-correct
    results), but re-sort the returned set by submission date — newer
    papers surface first, which matters for fast-moving research areas
    where older papers may already be superseded.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results * 2,  # fetch a slightly wider pool to sort from
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = list(client.results(search))

    # Most recently submitted/updated first
    results.sort(key=lambda r: r.updated, reverse=True)

    papers = []
    for result in results[:max_results]:
        papers.append(
            {
                "title": result.title,
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
            }
        )

    return papers