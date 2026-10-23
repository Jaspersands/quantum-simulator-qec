"""Independent inert parser assertions for intentional QDK report safety differences."""
import copy
import json
from html.parser import HTMLParser

import pytest

from stabilizer_qec.estimate.azure import EstimatorResult, LogicalCounts
from stabilizer_qec.estimate import azure_report as rep


class Parsed(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.starts, self.text = [], []
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.starts.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


MARKER = 'A & B <audit-marker data-note="inert"> C </audit-marker>'


def source():
    return {'status': 'success', 'value': 'ordinary', 'reportData': {'groups': [
        {'title': 'Group', 'alwaysVisible': True, 'entries': [
            {'label': 'Label', 'path': 'value', 'description': 'Description', 'explanation': 'Explanation'}]}], 'assumptions': ['Assumption']}}


def renderings(data):
    one = EstimatorResult(copy.deepcopy(data))
    batch = EstimatorResult([copy.deepcopy(data), copy.deepcopy(data)])
    return (one._repr_html_(), one.summary.content, batch._repr_html_(), batch[:].content)


@pytest.mark.parametrize('field', ['title', 'label', 'description', 'value'])
@pytest.mark.parametrize('markdown', [False, True])
def test_each_data_text_boundary(field, markdown, monkeypatch):
    if markdown and not rep.has_markdown:
        pytest.skip('optional markdown and mdx_math are absent')
    monkeypatch.setattr(rep, 'has_markdown', markdown)
    data = source()
    if field == 'value':
        data[field] = MARKER
    elif field == 'title':
        data['reportData']['groups'][0][field] = MARKER
    else:
        data['reportData']['groups'][0]['entries'][0][field] = MARKER
    original = copy.deepcopy(data)
    for index, rendered in enumerate(renderings(data)):
        parsed = Parsed(rendered)
        assert all(tag != 'audit-marker' and 'data-note' not in attrs for tag, attrs in parsed.starts)
        if field != 'description' or index < 2:  # batch tables do not show descriptions
            assert MARKER in ''.join(parsed.text)
    assert data == original
    assert renderings(data) == renderings(data)


@pytest.mark.parametrize('markdown', [False, True])
def test_rich_fields_are_data_with_fixed_formatting(markdown, monkeypatch):
    if markdown and not rep.has_markdown:
        pytest.skip('optional markdown and mdx_math are absent')
    monkeypatch.setattr(rep, 'has_markdown', markdown)
    rich = '**bold** _em_ `inline` [paper](https://example.invalid/paper?a=1&b=2) $Q_{\\rm alg}$ '
    rich += '<audit-marker data-note="inert">marker</audit-marker> <strong data-audit="inert">safe</strong>'
    data = source()
    data['reportData']['groups'][0]['entries'][0]['explanation'] = rich
    data['reportData']['assumptions'] = [rich]
    for index, rendered in enumerate(renderings(data)):
        parsed = Parsed(rendered)
        assert all(tag != 'audit-marker' and not ({'data-note', 'data-audit'} & set(attrs)) for tag, attrs in parsed.starts)
        if markdown:
            assert ('a', {'href': 'https://example.invalid/paper?a=1&b=2'}) in parsed.starts
            assert '$Q_{\\rm alg}$' in ''.join(parsed.text)
            assert {'strong', 'em', 'code'} <= {tag for tag, _ in parsed.starts}
        elif index < 2:
            assert rich in ''.join(parsed.text)
        else:
            assert '<li>' not in rendered  # assumptions are omitted without optional Markdown


@pytest.mark.parametrize('fragment', [
    '<a href="audit:marker">link</a>',
    '<a href="&#97;udit:marker">link</a>',
    '<a href=" javascript:marker">link</a>',
    '<a href="https://example.invalid/&#9;marker">link</a>',
    '<a href="//example.invalid/">link</a>',
    '<a href="http://[/">link</a>',
    '<a href="https:///missing-host">link</a>',
    '<a href="https://example.invalid:bad/">link</a>',
    '<script type="math/tex">synthetic math</script><style>synthetic style</style>',
    '<audit-marker/><img src="synthetic"><!-- comment --><!DOCTYPE audit>',
    '<p><strong>unclosed</p></strong>',
    '<![unknown[malformed declaration]]>',
])
def test_rich_fragment_policy_and_balance(fragment):
    # A controlled converter isolates serialization from optional Markdown versions. Expected
    # tags, href policy and balancing are asserted independently, never by calling its sanitizer.
    class Converter:
        def convert(self, value):
            return value
    result = rep._report_rich_text(fragment, Converter())
    parsed = Parsed(result)
    assert all(tag in {'p', 'em', 'strong', 'code', 'pre', 'br', 'ul', 'ol', 'li', 'blockquote', 'a'} for tag, _ in parsed.starts)
    assert all(set(attrs) <= {'href', 'title'} if tag == 'a' else not attrs for tag, attrs in parsed.starts)
    assert all('href' not in attrs for _, attrs in parsed.starts)

    class Balanced(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
        def handle_starttag(self, tag, attrs):
            if tag != 'br':
                self.stack.append(tag)
        def handle_endtag(self, tag):
            assert self.stack and self.stack.pop() == tag
    balanced = Balanced()
    balanced.feed(result)
    assert not balanced.stack


def test_default_math_assumption_and_https_formatting():
    if not rep.has_markdown:
        pytest.skip('optional markdown and mdx_math are absent')
    result = LogicalCounts(num_qubits=12, t_count=500).estimate()
    parsed = Parsed(result._repr_html_())
    assert any(tag == 'a' and attrs.get('href', '').startswith('https://') for tag, attrs in parsed.starts)
    assert any(tag == 'strong' for tag, _ in parsed.starts)
    assert '$Q_{\\rm alg}' in ''.join(parsed.text)


def test_diagram_transport_preserves_strings_and_object():
    data = source()
    data['audit_label'] = 'Model A & B "quoted" \'single\' > <audit-marker> </script> \\ backslash\t\n Unicode é'
    original = copy.deepcopy(data)
    expected = {k: v for k, v in original.items() if k != 'reportData'}
    result = EstimatorResult(data)
    for kind in ('space', 'time'):
        rendered = getattr(result.diagram, kind).content
        parsed = Parsed(rendered)
        assert [tag for tag, _ in parsed.starts] == ['script', 're-' + kind + '-diagram']
        assert set(parsed.starts[0][1]) == {'src'}
        assert set(parsed.starts[1][1]) == {'data'}
        assert json.loads(parsed.starts[1][1]['data']) == expected
        assert getattr(result.diagram, kind).content == rendered
    assert json.loads(result.diagram.data_json) == expected and data == original


def test_real_markdown_links_are_filtered():
    if not rep.has_markdown:
        pytest.skip('optional markdown and mdx_math are absent')
    data = source()
    data['reportData']['groups'][0]['entries'][0]['explanation'] = '[marker](audit:marker) <a href="&#97;udit:marker">entity</a>'
    parsed = Parsed(EstimatorResult(data)._repr_html_())
    assert all('href' not in attrs for tag, attrs in parsed.starts if tag == 'a')
    assert 'marker' in ''.join(parsed.text) and 'entity' in ''.join(parsed.text)


def test_direct_diagram_constructor_preserves_caller_object():
    data = source()
    original = copy.deepcopy(data)
    rep.EstimatorResultDiagram(data)
    assert data == original
