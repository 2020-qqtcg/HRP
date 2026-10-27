"""Versioned, lossy intent projections for public terminal actions.

Neither shared intent nor shared syntax establishes behavioral equivalence.
V1 is frozen for paired comparisons. V2 preserves static structure and uncertainty.
"""
from __future__ import annotations

import hashlib
import re
import shlex
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_bash

TERMINAL_ACTIONS = ("inspect", "modify", "verify", "run", "setup", "opaque", "stop")
CLASSIFIER_VERSION = "bash-ast-v2"
_LEGACY_VERSION = "dominant-v1"
_BASH_LANGUAGE = Language(tree_sitter_bash.language())

_INSPECT = {
    "awk",
    "basename",
    "cat",
    "comm",
    "cut",
    "df",
    "diff",
    "dirname",
    "du",
    "env",
    "file",
    "find",
    "free",
    "grep",
    "head",
    "id",
    "jq",
    "less",
    "locate",
    "ls",
    "lsof",
    "md5sum",
    "nl",
    "od",
    "pgrep",
    "printenv",
    "ps",
    "pwd",
    "readelf",
    "realpath",
    "rg",
    "sha256sum",
    "sort",
    "stat",
    "strings",
    "tail",
    "tree",
    "type",
    "uniq",
    "wc",
    "which",
}
_MODIFY = {
    "chmod",
    "chown",
    "cp",
    "dd",
    "install",
    "ln",
    "mkdir",
    "mkfifo",
    "mknod",
    "mv",
    "patch",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}
_VERIFY = {
    "bats",
    "jest",
    "mypy",
    "nose2",
    "phpunit",
    "pytest",
    "ruff",
    "shellcheck",
    "tox",
    "vitest",
}
_SETUP = {
    "apt",
    "apt-get",
    "brew",
    "conda",
    "curl",
    "dnf",
    "dpkg",
    "gem",
    "pacman",
    "pip",
    "pip3",
    "service",
    "systemctl",
    "uv",
    "wget",
    "yum",
}
_RUN = {
    "bash",
    "cargo",
    "cmake",
    "gcc",
    "g++",
    "go",
    "java",
    "make",
    "node",
    "perl",
    "php",
    "python",
    "python3",
    "ruby",
    "sh",
}
_NEUTRAL = {".", "cd", "export", "set", "source", "umask", "unset"}
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_TEST_WORD = re.compile(r"(^|[-_/.])(check|lint|spec|test|tests)([-_/.]|$)")
_ACTION_DOMINANCE = {
    "opaque": 0,
    "inspect": 1,
    "run": 2,
    "verify": 3,
    "setup": 4,
    "modify": 5,
}


def _dominant_semantic_action(categories: list[str]) -> str:
    recognized = [category for category in categories if category != "opaque"]
    candidates = recognized or categories or ["opaque"]
    return max(
        enumerate(candidates),
        key=lambda item: (_ACTION_DOMINANCE[item[1]], item[0]),
    )[1]


def _shell_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(
        command.replace("\n", " ; "),
        posix=True,
        punctuation_chars=";&|<>()",
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in {"&&", "||", ";", "|", "&", "(", ")"}:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _unwrap_command(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    while tokens and (_ASSIGNMENT.match(tokens[0]) or tokens[0] in {"sudo", "command"}):
        tokens.pop(0)
    if tokens and tokens[0] == "env":
        tokens.pop(0)
        while tokens and (tokens[0].startswith("-") or _ASSIGNMENT.match(tokens[0])):
            tokens.pop(0)
    if tokens and tokens[0] in {"nohup", "time"}:
        tokens.pop(0)
    if tokens and tokens[0] == "timeout":
        tokens.pop(0)
        while tokens and tokens[0].startswith("-"):
            tokens.pop(0)
        if tokens:
            tokens.pop(0)
    return tokens


def _classify_segment(tokens: list[str]) -> str | None:
    tokens = _unwrap_command(tokens)
    if not tokens:
        return None

    executable = Path(tokens[0]).name.lower()
    arguments = [token.lower() for token in tokens[1:]]
    joined = " ".join(arguments)

    if executable in _NEUTRAL:
        return None
    if executable == "git":
        subcommand = next((item for item in arguments if not item.startswith("-")), "")
        if subcommand in {"status", "diff", "log", "show", "rev-parse", "branch", "grep"}:
            return "inspect"
        return "modify"
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        if any(item in arguments for item in {"install", "add", "ci", "update"}):
            return "setup"
        if any(_TEST_WORD.search(item) for item in arguments):
            return "verify"
        return "run"
    if executable in {"python", "python3"} and len(arguments) >= 2:
        if arguments[0] == "-m" and arguments[1] in _VERIFY:
            return "verify"
        if arguments[0] in {"-c", "-"}:
            return "opaque"
    if executable in {"bash", "sh"} and len(tokens) >= 3 and tokens[1] == "-c":
        return classify_bash_command_v1(tokens[2])
    if executable == "find" and any(item in arguments for item in {"-delete", "-exec", "-execdir"}):
        return "modify"
    if executable in {"sed", "perl"} and any(
        item == "-i" or item.startswith("-i") for item in arguments
    ):
        return "modify"
    if executable in {"echo", "printf", "cat"} and any(
        token in {">", ">>", ">|"} for token in tokens
    ):
        return "modify"
    if executable in _VERIFY or _TEST_WORD.search(executable):
        return "verify"
    if executable in _INSPECT:
        return "inspect"
    if executable in _MODIFY:
        return "modify"
    if executable in _SETUP:
        return "setup"
    if executable == "make":
        if _TEST_WORD.search(joined):
            return "verify"
        if any(item in arguments for item in {"install", "bootstrap", "deps"}):
            return "setup"
        return "run"
    if executable == "cargo":
        return "verify" if "test" in arguments else "run"
    if executable == "go":
        return "verify" if "test" in arguments else "run"
    if executable in _RUN or executable.startswith("./"):
        return "run"
    if "/" in tokens[0] or tokens[0].startswith("."):
        return "verify" if _TEST_WORD.search(tokens[0].lower()) else "run"
    return "opaque"


def classify_bash_command_v1(command: str) -> str:
    """Project a concrete Bash program onto the finite Terminal HRP vocabulary.

    Each shell clause is classified independently, then projected to the
    dominant observable effect. Persistent task changes dominate setup,
    verification, execution, and inspection; recognized effects dominate
    opaque helper clauses. The exact Bash program remains unchanged in the
    corresponding :class:`BashToolCall`.
    """

    if not isinstance(command, str) or not command.strip():
        raise ValueError("bash command must be a non-empty string")
    try:
        categories = [
            category
            for segment in _shell_segments(command)
            if (category := _classify_segment(segment)) is not None
        ]
    except ValueError:
        return "opaque"
    return _dominant_semantic_action(categories)


def _literal(node: Any) -> str:
    """Unquote one AST word; never lex a heredoc or an entire shell program."""
    raw = node.text.decode("utf-8")
    try:
        words = shlex.split(raw)
        return words[0] if len(words) == 1 else raw
    except ValueError:
        return raw


def _unwrap_v2(words: list[str]) -> tuple[list[str], list[str]]:
    wrappers: list[str] = []
    words = list(words)
    while words:
        exe = Path(words[0]).name
        if exe == "command" and any(x in {"-v", "-V"} for x in words[1:]):
            break
        if exe in {"command", "builtin", "exec", "nohup", "sudo", "env", "timeout", "time"}:
            wrappers.append(words.pop(0))
            # Options with separate operands must not become the executable.
            valued = {
                "sudo": {"-u", "-g", "-h", "-p", "-C", "-T"},
                "env": {"-u", "--unset", "-C", "--chdir"},
                "timeout": {"-k", "--kill-after", "-s", "--signal"},
                "time": {"-f", "--format", "-o", "--output"},
            }.get(exe, set())
            while words and (words[0].startswith("-") or _ASSIGNMENT.match(words[0])):
                option = words.pop(0)
                wrappers.append(option)
                if option in valued and words:
                    wrappers.append(words.pop(0))
                if option == "--":
                    break
            if exe == "timeout" and words:
                wrappers.append(words.pop(0))
        elif exe == "uv" and len(words) > 1 and words[1] == "run":
            wrappers.extend(words[:2])
            words = words[2:]
            while words and words[0].startswith("-"):
                option = words.pop(0)
                wrappers.append(option)
                if option in {"--with", "--with-editable", "--with-requirements", "--python", "--project", "--directory", "--group", "--extra", "--package"} and words:
                    wrappers.append(words.pop(0))
                if option == "--":
                    break
        else:
            break
    return words, wrappers


def _intent(words: list[str]) -> tuple[str | None, str, list[str]]:
    """Conservative intent rules. The last component records unresolved effects."""
    if not words:
        return "opaque", "missing_program", ["missing_program"]
    exe, args = Path(words[0]).name.lower(), words[1:]
    if exe in {"true", "false", ":"}:
        # Constant statuses affect shell control flow, not task verification.
        # Keep the operation; the AST walk still records argument effects and
        # redirections. Do not add these to _NEUTRAL, which is shared with V1.
        return None, "constant_exit_status", []
    if exe in _NEUTRAL:
        return None, "shell_context", (["sourced_program"] if exe in {".", "source"} else [])
    if exe in {"bash", "sh"}:
        return "run", "shell_program", ["external_shell_program"]
    if exe == "command" and any(x in {"-v", "-V"} for x in args):
        return "inspect", "command_lookup", []
    if exe == "git":
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i] in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"} else 1
        sub, rest = (args[i], args[i + 1:]) if i < len(args) else ("", [])
        if sub in {"status", "diff", "log", "show", "rev-parse", "grep", "ls-files", "ls-tree", "describe"}:
            return "inspect", f"git_{sub}", []
        if sub in {"branch", "tag"}:
            read = not rest or any(x in {"--list", "-l", "--show-current", "--contains", "--merged", "--no-merged"} for x in rest) or all(x in {"-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose"} for x in rest)
            mutating = any(x in {"-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy", "-f", "--force", "--edit-description"} for x in rest)
            return ("inspect" if read and not mutating else "modify"), f"git_{sub}", []
        if sub in {"add", "apply", "am", "commit", "checkout", "switch", "restore", "reset", "clean", "merge", "rebase", "cherry-pick", "revert", "rm", "mv", "stash", "clone", "fetch", "pull", "push", "init"}:
            return "modify", f"git_{sub}", []
        return "opaque", f"git_{sub or 'unknown'}", ["unresolved_git_subcommand"]
    if exe == "find":
        if "-delete" in args or any(x in args for x in {"-fprint", "-fprint0", "-fprintf"}):
            return "modify", "find_write", []
        return "inspect", "find_read", []
    if exe in {"python", "python3", "python3.12", "python3.11", "python3.13"}:
        if "-m" in args:
            i = args.index("-m") + 1
            module = args[i] if i < len(args) else ""
            if module in _VERIFY or module == "unittest":
                return "verify", "python_test_module", []
            if module in {"pip", "venv", "ensurepip"}:
                return "setup", "python_setup_module", []
            return "run", "python_module", ["unresolved_program_effects"]
        if not args or "-c" in args or "-" in args:
            return "opaque", "embedded_python", ["embedded_program"]
        return "run", "python_script", ["unresolved_program_effects"]
    if exe in {"sed", "perl"} and any(
        x.startswith("--in-place") or re.match(r"^-[A-Za-z]*i", x) for x in args
    ):
        return "modify", "in_place_edit", []
    if exe == "sed":
        return "inspect", "stream_transform", ["sed_script_effects"]
    if exe in {"curl", "wget"}:
        writes = exe == "wget" or any(x in {"-o", "-O", "--output", "--output-document", "--remote-name"} or x.startswith(("--output=", "-o")) for x in args)
        return ("setup" if writes else "inspect"), "network_transfer", ["remote_effects"]
    if exe in {"echo", "printf"}:
        return "inspect", "emit_text", []
    if exe in _VERIFY:
        return "verify", "test_or_lint", []
    if exe in _INSPECT:
        return "inspect", "read", (["embedded_program"] if exe in {"awk", "jq"} else [])
    if exe in _MODIFY:
        return "modify", {"rm": "remove", "mv": "move", "cp": "copy", "mkdir": "create_directory", "chmod": "permissions"}.get(exe, "filesystem_write"), []
    if exe in _SETUP:
        return "setup", "environment_setup", []
    if exe in {"make", "cargo", "go", "npm", "pnpm", "yarn", "bun"}:
        if any(x in {"test", "tests", "check", "lint", "spec", "clippy"} for x in args):
            return "verify", "test_target", []
        if any(x in {"install", "add", "ci", "update", "bootstrap", "deps"} for x in args):
            return "setup", "install_target", []
        return "run", "build_or_script", ["unresolved_program_effects"]
    if exe in {"test", "[", "[["}:
        return "verify", "shell_predicate", []
    if exe in _RUN or "/" in words[0] or words[0].startswith("."):
        return "run", "program", ["unresolved_program_effects"]
    return "opaque", "unknown_program", ["unknown_program"]


def analyze_bash_command(command: str) -> dict[str, Any]:
    """Describe static syntax in source order, not an asserted execution trace.

    The primary label is one lossy observation per decision. Auxiliary operations
    never become extra HRP samples. Unknown fragments remain visible even when
    another operation determines the primary label.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("bash command must be a non-empty string")
    operations: list[dict[str, Any]] = []
    structure: list[dict[str, Any]] = []
    reasons: list[str] = []
    root = Parser(_BASH_LANGUAGE).parse(command.encode()).root_node

    def add(node: Any, words: list[str], context: list[str]) -> None:
        words, wrappers = _unwrap_v2(words)
        category, subtype, unknown = _intent(words)
        if "uv" in wrappers:
            unknown.append("wrapper_may_install_dependencies")
        op = {
            "category": category, "subtype": subtype,
            "program": words[0] if words else "", "arguments": words[1:],
            "objects": [x for x in words[1:] if not x.startswith("-")],
            "wrappers": wrappers, "span": [node.start_byte, node.end_byte],
            "context": context, "unknown_reasons": unknown,
        }
        operations.append(op)
        # Nested interpreters are only expanded when -c has a literal argument.
        if words and Path(words[0]).name in {"bash", "sh"} and len(words) >= 3 and words[1] in {"-c", "-lc", "-ec", "-euc"}:
            dynamic = any(n.type in {"expansion", "simple_expansion", "command_substitution"} for n in descendants(node))
            if not dynamic:
                nested = analyze_bash_command(words[2])
                op.update(category=nested["semantic_action"], subtype="inline_shell", nested=nested)
                op["unknown_reasons"] = nested["unknown_reasons"]
        if words and Path(words[0]).name == "find":
            for marker_index, marker in enumerate(words):
                if marker not in {"-exec", "-execdir", "-ok", "-okdir"}:
                    continue
                start = marker_index + 1
                end = next((i for i in range(start, len(words)) if words[i] in {";", "+"}), len(words))
                add(node, words[start:end], context + [marker])
        reasons.extend(op["unknown_reasons"])

    def descendants(node: Any):
        for child in node.named_children:
            yield child
            yield from descendants(child)

    def walk(node: Any, context: list[str]) -> None:
        if node.type == "comment":
            return
        if node.type == "heredoc_body":
            # Unquoted heredocs may expand substitutions; literal text is data.
            for child in node.named_children:
                if child.type in {"expansion", "simple_expansion", "command_substitution"}:
                    reasons.append("dynamic_heredoc")
                    walk(child, context + ["heredoc_expansion"])
            return
        if node.type == "function_definition":
            structure.append({"type": node.type, "span": [node.start_byte, node.end_byte]})
            reasons.append("function_definition_not_executed")
            return
        if node.type in {"list", "pipeline", "if_statement", "for_statement", "while_statement", "case_statement", "subshell", "command_substitution"}:
            structure.append({"type": node.type, "span": [node.start_byte, node.end_byte], "operators": [c.type for c in node.children if not c.is_named]})
            context = context + [node.type]
        if node.type == "test_command":
            # Tree-sitter represents [ ... ] and [[ ... ]] separately from
            # ordinary commands; continue walking their substitutions below.
            add(node, [_literal(node.children[0]), *(_literal(arg) for arg in node.named_children)], context)
        if node.type == "command":
            name = node.child_by_field_name("name")
            args = node.children_by_field_name("argument")
            if name:
                operation_index = len(operations)
                add(node, [_literal(name), *(_literal(a) for a in args)], context)
                if any(n.type in {"expansion", "simple_expansion", "command_substitution"} for n in descendants(node)):
                    reasons.append("dynamic_expansion")
                if name.type != "command_name" or any(n.type in {"expansion", "simple_expansion", "command_substitution"} for n in descendants(name)):
                    operations[operation_index]["category"] = "opaque"
                    reasons.append("dynamic_program")
            # Traverse only substitutions/redirections, not literal script bodies.
            for child in node.named_children:
                walk(child, context)
            return
        if node.type == "file_redirect":
            dest = node.child_by_field_name("destination")
            raw = node.text.decode()
            operator = next((c.type for c in node.children if not c.is_named and c.type != "file_descriptor"), "")
            target = _literal(dest) if dest else ""
            writes = operator in {">", ">>", ">|", "&>", "&>>", "<>"}
            fd_only = operator in {">&", "<&"} and (target.isdigit() or target == "-")
            if not fd_only and (writes or operator == ">&") and target not in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
                operations.append({"category": "modify", "subtype": "append" if ">>" in operator else "redirect_write", "program": "", "arguments": [raw], "objects": [target], "wrappers": [], "span": [node.start_byte, node.end_byte], "context": context, "unknown_reasons": []})
        for child in node.named_children:
            walk(child, context)

    walk(root, [])
    if root.has_error:
        reasons.append("bash_parse_error")
    categories = [op["category"] for op in operations if op["category"]]
    return {
        "version": CLASSIFIER_VERSION,
        "fingerprint": CLASSIFIER_FINGERPRINT,
        "semantic_action": "opaque" if root.has_error else _dominant_semantic_action(categories),
        "legacy_semantic_action": classify_bash_command_v1(command),
        "parse_status": "error" if root.has_error else "ok",
        "unknown_reasons": sorted(set(reasons)),
        "operations": operations,
        "objects_are": "unresolved_syntactic_arguments_not_execution_effects",
        "structure": structure,
    }


def classify_bash_command(command: str) -> str:
    return analyze_bash_command(command)["semantic_action"]


# Changing rules creates a separate checkpoint namespace automatically.
PARSER_VERSIONS = {package: version(package) for package in ("tree-sitter", "tree-sitter-bash")}
CLASSIFIER_FINGERPRINT = hashlib.sha256(
    Path(__file__).read_bytes() + repr(sorted(PARSER_VERSIONS.items())).encode()
).hexdigest()
TERMINAL_ACTION_PROJECTION = f"{CLASSIFIER_VERSION}-{CLASSIFIER_FINGERPRINT[:12]}"
