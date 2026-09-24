import re

from graph.textsafe import safe_text


def test_markdown_image_and_link_are_neutralised():
    out = safe_text("![x](http://evil/?d=1) and [a](http://b)")
    assert "![" not in out and "](" not in out
    assert "http://evil/?d=1" in out.replace("\\", "")   # still readable, just literal


def test_html_and_control_characters_are_escaped():
    out = safe_text("<img src=x onerror=alert(1)> `code` *b* _i_ # h | t")
    for ch in "<>`*_#|":
        assert not re.search(r"(?<!\\)" + re.escape(ch), out), ch


def test_leading_list_and_heading_markers_are_escaped():
    out = safe_text("- item\n1. one\n+ plus\n# head")
    assert out.split("\n") == [r"\- item", r"1\. one", r"\+ plus", r"\# head"]


def test_plain_text_and_non_strings_pass_through_readably():
    assert safe_text("Engineering 20 percent") == "Engineering 20 percent"
    assert safe_text(None) == ""
    assert safe_text(5) == "5"
