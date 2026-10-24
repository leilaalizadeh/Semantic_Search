"""The code in src/semantic_search/levels/ must stay a faithful copy of the git tags.

Comments, note strings and blank lines may differ (you can add your own notes), the code may not.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {"level1": "Level1", "level2": "level2", "level3": "level3"}  # folder -> git tag


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def tag_exists(tag: str) -> bool:
    try:
        git("rev-parse", "--verify", f"refs/tags/{tag}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def allowed_changes(code: str, folder: str) -> str:
    """The only two edits made when copying: import paths and data-folder depth."""
    code = re.sub(r"from semantic_search\.([a-z_]+) import", rf"from semantic_search.levels.{folder}.\1 import", code)
    code = code.replace("from semantic_search import", f"from semantic_search.levels.{folder} import")
    return code.replace("parents[2]", "parents[4]")


class RemoveNotes(ast.NodeTransformer):
    """Drop standalone strings ('''notes''' and docstrings): they don't change what the code does."""

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def code_only(source: str) -> str:
    # ast.dump = the code as Python understands it, without # comments or formatting
    return ast.dump(RemoveNotes().visit(ast.parse(source)))


@pytest.mark.parametrize("folder, tag", LEVELS.items())
def test_level_matches_git_tag(folder, tag):
    if not tag_exists(tag):
        pytest.skip(f"git tag {tag} not available")
    files = git("ls-tree", "-r", "--name-only", tag, "src/semantic_search/").split()
    for path in files:
        original = git("show", f"{tag}:{path}")
        copy = (ROOT / "src/semantic_search/levels" / folder / Path(path).name).read_text()
        same_code = code_only(copy) == code_only(allowed_changes(original, folder))
        assert same_code, f"{folder}/{Path(path).name}: code differs from tag {tag}"


def test_level1_code_still_works():
    from semantic_search.levels import level1

    engine = level1.SemanticSearch(level1.load_products())
    titles = [p.title for _, p in engine.search("gift for a 6 year old", top_k=2)]
    assert any("LEGO" in t for t in titles)
