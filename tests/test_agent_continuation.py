"""Agent-loop regressions: current schemas, genuine tool calls, no fake end-turn."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from freetokenapi import tools as tool_api

TOOLS = [
    {"type": "function", "function": {"name": name, "description": "Local client operation", "parameters": {
        "type": "object", "properties": properties, "required": list(properties),
    }}}
    for name, properties in [
        ("Bash", {"command": {"type": "string"}, "description": {"type": "string"}}),
        ("Write", {"file_path": {"type": "string"}, "content": {"type": "string"}}),
        ("Read", {"file_path": {"type": "string"}}),
    ]
]
SCHEMAS = tool_api.tool_schema_map(TOOLS)


def msg(role, content="", **kwargs):
    return SimpleNamespace(role=role, content=content, **kwargs)


def history():
    return [
        msg("user", "Create the page, save it locally, and verify the result."),
        msg("assistant", tool_calls=[{"id": "a", "function": {"name": "Read", "arguments": '{"file_path":"source.txt"}'}}]),
        msg("tool", "OLD_READ_RESULT", tool_call_id="a"),
        msg("assistant", tool_calls=[{"id": "b", "function": {"name": "Write", "arguments": '{"file_path":"page.html","content":"ok"}'}}]),
        msg("tool", "NEW_WRITE_RESULT", tool_call_id="b"),
        msg("tool", "SECOND_CURRENT_RESULT", tool_call_id="b2"),
    ]


@pytest.mark.parametrize("has_session", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", {"type": "function", "function": {"name": "Read"}}])
def test_each_tool_round_has_current_schema_and_execution_contract(has_session, choice):
    prompt, mode = tool_api.build_prompt(history(), TOOLS, choice, has_session=has_session)
    assert mode
    assert '"file_path":{"type":"string"}' in prompt
    assert tool_api.AGENT_EXECUTION_RULES in prompt
    assert "provide the final answer based on the tool results" not in prompt
    if choice == "required":
        assert "You MUST call one or more functions" in prompt
    elif isinstance(choice, dict):
        assert "You MUST call exactly the function Read" in prompt
    if has_session:
        assert "OLD_READ_RESULT" not in prompt
        assert "NEW_WRITE_RESULT" in prompt and "SECOND_CURRENT_RESULT" in prompt


def test_new_user_turn_refreshes_changed_tool_schema_without_replaying_history():
    messages = history() + [msg("assistant", "Saved."), msg("user", "Now verify it.")]
    prompt, mode = tool_api.build_prompt(messages, [TOOLS[2]], has_session=True)
    assert mode and "Read" in prompt and "file_path" in prompt
    assert "Write" not in prompt and "OLD_READ_RESULT" not in prompt
    assert "Now verify it." in prompt


@pytest.mark.parametrize("has_session", [False, True])
def test_disabled_tools_do_not_reactivate_from_history(has_session):
    prompt, mode = tool_api.build_prompt(history(), TOOLS, "none", has_session=has_session)
    assert not mode
    assert tool_api.NO_CLIENT_TOOLS in prompt
    assert "Available functions:" not in prompt


@pytest.mark.parametrize("reply", [
    '[assistant called Bash({"command":"python check.py","description":"check"})]',
    '[调用 Bash] {"command":"python check.py","description":"check"}',
    '[Call Bash] {"command":"python check.py","description":"check"}',
])
def test_observed_claude_call_notation_is_not_plain_text(reply):
    calls, text = tool_api.parse_tool_calls(reply, SCHEMAS)
    assert len(calls) == 1 and calls[0].name == "Bash" and not text
    assert json.loads(calls[0].arguments)["command"] == "python check.py"


def test_multiple_bracket_calls_preserve_arguments_and_surrounding_text():
    args = {"file_path": "C:\\工作 目录\\page.html", "content": "line 1\n[调用 Bash] is just file content\n</script>"}
    reply = "I will save and verify it.\n[调用 Write] " + json.dumps(args, ensure_ascii=False)
    reply += '\n[调用 Read] {"file_path":"page.html"}\nChecking the written file.'
    calls, text = tool_api.parse_tool_calls(reply, SCHEMAS)
    assert [call.name for call in calls] == ["Write", "Read"]
    assert json.loads(calls[0].arguments) == args
    assert "I will save" in text and "Checking" in text


@pytest.mark.parametrize("reply", [
    '[调用 PowerShell] {"command":"Get-Item x"}',
    '[调用 Bash] {"command":"truncated',
    'Example: [调用 Bash] {"command":"do not run"}',
    'Previously: [assistant called Bash({"command":"do not repeat"})]',
    '[assistant called Bash({"command":"old"})] was the previous operation.',
    '\x60\x60\x60text\n[调用 Bash] {"command":"example"}\n\x60\x60\x60',
    '\x60[调用 Bash] {"command":"example"}\x60',
    '> [调用 Bash] {"command":"quoted"}',
])
def test_bracket_examples_unknown_names_and_truncation_are_not_executed(reply):
    reply = reply.replace("\\x60", chr(96))
    assert tool_api.parse_tool_calls(reply, SCHEMAS) is None


def test_bracket_calls_require_declared_tools():
    assert tool_api.parse_tool_calls('[调用 Bash] {"command":"x"}') is None


def test_history_uses_a_parseable_canonical_call_not_prose():
    rendered = tool_api.render_message(history()[1])
    assert "assistant called" not in rendered
    calls, _ = tool_api.parse_tool_calls(rendered, SCHEMAS)
    assert calls[0].name == "Read"


@pytest.mark.parametrize("content", ['{"nonce":"abc","stage":"complete"}', '[1,2,3]', 'true', '42'])
def test_xml_json_looking_file_contents_remain_strings(content):
    reply = '<tool_calls><invoke name="Write"><parameter name="file_path">report.json</parameter><parameter name="content">' + content + '</parameter></invoke></tool_calls>'
    calls, _ = tool_api.parse_tool_calls(reply, SCHEMAS)
    assert json.loads(calls[0].arguments)["content"] == content


@pytest.mark.parametrize("encoded,expected", [
    ('<![CDATA[{"nonce":"abc"}\n]]>', '{"nonce":"abc"}\n'),
    ('<![CDATA[  <tag>&amp;</tag>\n]]>', '  <tag>&amp;</tag>\n'),
    ('<![CDATA[a]]]]><![CDATA[>b]]>', 'a]]>b'),
])
def test_xml_cdata_does_not_corrupt_written_files(encoded, expected):
    reply = '<tool_calls><invoke name="Write"><parameter name="file_path">report.json</parameter><parameter name="content">' + encoded + '</parameter></invoke></tool_calls>'
    calls, _ = tool_api.parse_tool_calls(reply, SCHEMAS)
    assert json.loads(calls[0].arguments)["content"] == expected


def test_actual_array_arguments_are_still_arrays():
    schemas = {"edit": {"path": "string", "edits": "array"}}
    reply = '<tool_calls><invoke name="edit"><parameter name="path">source.txt</parameter><parameter name="edits">[{"oldText":"old","newText":"new"}]</parameter></invoke></tool_calls>'
    calls, _ = tool_api.parse_tool_calls(reply, schemas)
    assert json.loads(calls[0].arguments)["edits"] == [{"oldText": "old", "newText": "new"}]
