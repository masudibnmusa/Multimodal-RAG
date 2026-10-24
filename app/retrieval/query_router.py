import re

VISUAL_RE = re.compile(
    r"\b(charts?|graphs?|plots?|bars?|pie|histograms?|tables?|rows?|columns?|diagrams?|figures?|"
    r"architecture|flowcharts?|arrows?|axis|axes|legend|trend|connects?|illustrat\w*|shown|depicted)\b",
    re.IGNORECASE,
)


def route_query(query: str) -> dict:
    """Cheap heuristic: does the question target visual content?"""
    return {"visual": bool(VISUAL_RE.search(query))}