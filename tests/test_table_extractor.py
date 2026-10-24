from app.parsing.table_extractor import clean_markdown


def test_strips_code_fences():
    raw = "```markdown\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
    assert clean_markdown(raw) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_plain_table_unchanged():
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert clean_markdown(table) == table