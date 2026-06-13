"""
Tests for .coderabbit.yaml — the CodeRabbit AI code-review configuration.

These tests verify that the configuration file:
  - Is valid, parseable YAML
  - Contains all required top-level sections
  - Has correctly typed field values
  - Encodes the five architectural constraints that are central to the
    modelinfo-cli project in the path_instructions review prompt
"""

import os
import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to test the CodeRabbit config")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", ".coderabbit.yaml")


@pytest.fixture(scope="module")
def config():
    """Parse .coderabbit.yaml once and share across all tests in this module."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def path_instruction(config):
    """Return the first (and currently only) path_instructions entry."""
    return config["reviews"]["path_instructions"][0]


# ---------------------------------------------------------------------------
# File-level validity
# ---------------------------------------------------------------------------

def test_yaml_file_exists():
    """The .coderabbit.yaml file must exist at the repository root."""
    assert os.path.isfile(CONFIG_PATH), f"Config file not found at {CONFIG_PATH}"


def test_yaml_is_valid_and_non_empty(config):
    """The file must parse into a non-empty dict, not None or a scalar."""
    assert isinstance(config, dict), "Top-level YAML document must be a mapping"
    assert len(config) > 0, "Config must not be empty"


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------

def test_top_level_language_key_present(config):
    assert "language" in config


def test_top_level_reviews_key_present(config):
    assert "reviews" in config


def test_top_level_chat_key_present(config):
    assert "chat" in config


def test_language_value(config):
    """Language must be set to US English."""
    assert config["language"] == "en-US"


# ---------------------------------------------------------------------------
# reviews section — scalar fields
# ---------------------------------------------------------------------------

def test_reviews_is_mapping(config):
    assert isinstance(config["reviews"], dict)


def test_reviews_profile_is_chill(config):
    """The 'chill' profile keeps review comments non-blocking and friendly."""
    assert config["reviews"]["profile"] == "chill"


def test_request_changes_workflow_is_false(config):
    """request_changes_workflow should be disabled so PRs are not hard-blocked."""
    assert config["reviews"]["request_changes_workflow"] is False


def test_high_level_summary_is_true(config):
    """A high-level summary should be generated for every PR."""
    assert config["reviews"]["high_level_summary"] is True


def test_poem_is_false(config):
    """The decorative poem feature should be turned off."""
    assert config["reviews"]["poem"] is False


def test_review_status_is_true(config):
    """Review status updates should be enabled."""
    assert config["reviews"]["review_status"] is True


def test_collapse_walkthrough_is_false(config):
    """Walkthrough comments should be expanded (not collapsed) by default."""
    assert config["reviews"]["collapse_walkthrough"] is False


# ---------------------------------------------------------------------------
# reviews.path_instructions structure
# ---------------------------------------------------------------------------

def test_path_instructions_is_list(config):
    assert isinstance(config["reviews"]["path_instructions"], list)


def test_path_instructions_is_non_empty(config):
    assert len(config["reviews"]["path_instructions"]) >= 1


def test_path_instruction_has_path_key(path_instruction):
    assert "path" in path_instruction


def test_path_instruction_has_instructions_key(path_instruction):
    assert "instructions" in path_instruction


def test_path_instruction_glob_is_catch_all(path_instruction):
    """The path glob must match every file in the repository."""
    assert path_instruction["path"] == "**/*"


def test_path_instruction_instructions_is_string(path_instruction):
    assert isinstance(path_instruction["instructions"], str)


def test_path_instruction_instructions_is_non_empty(path_instruction):
    assert len(path_instruction["instructions"].strip()) > 0


def test_path_instruction_references_project_name(path_instruction):
    """Instructions must identify the project so reviewers have context."""
    assert "modelinfo-cli" in path_instruction["instructions"]


# ---------------------------------------------------------------------------
# Architectural constraint #1 — Zero-Dependency Parsing
# ---------------------------------------------------------------------------

def test_instructions_enforces_zero_dependency_parsing(path_instruction):
    """Constraint: binary headers parsed with stdlib only (no third-party parsers)."""
    text = path_instruction["instructions"]
    assert "Zero-Dependency" in text or "zero-dependency" in text or "zero dependency" in text.lower()


def test_instructions_names_allowed_stdlib_modules(path_instruction):
    """The constraint must list the approved stdlib modules explicitly."""
    text = path_instruction["instructions"]
    for module in ("os", "struct", "json", "zipfile"):
        assert module in text, f"Expected stdlib module '{module}' to be listed in instructions"


def test_instructions_names_target_file_formats(path_instruction):
    """The constraint must mention the binary formats the project handles."""
    text = path_instruction["instructions"]
    assert ".safetensors" in text
    assert ".gguf" in text


# ---------------------------------------------------------------------------
# Architectural constraint #2 — No Heavy Imports
# ---------------------------------------------------------------------------

def test_instructions_bans_heavy_imports(path_instruction):
    """Constraint: heavy ML libraries must be explicitly forbidden."""
    text = path_instruction["instructions"]
    assert "No Heavy Imports" in text or "banned" in text.lower()


def test_instructions_lists_all_banned_libraries(path_instruction):
    """Every banned library must be named so reviewers know what to flag."""
    text = path_instruction["instructions"]
    for lib in ("torch", "transformers", "numpy", "safetensors", "huggingface_hub"):
        assert lib in text, f"Banned library '{lib}' not mentioned in instructions"


# ---------------------------------------------------------------------------
# Architectural constraint #3 — Fast Startup Time
# ---------------------------------------------------------------------------

def test_instructions_enforces_startup_time_limit(path_instruction):
    """Constraint: CLI must start in under 100ms."""
    text = path_instruction["instructions"]
    assert "100ms" in text or "100 ms" in text


def test_instructions_requires_lazy_imports(path_instruction):
    """Constraint: slow modules must be lazy-imported."""
    text = path_instruction["instructions"]
    assert "lazy" in text.lower() or "lazy-import" in text.lower()


# ---------------------------------------------------------------------------
# Architectural constraint #4 — Sandboxed Pickling
# ---------------------------------------------------------------------------

def test_instructions_prohibits_bare_pickle_load(path_instruction):
    """Constraint: pickle.load() must not be called without a sandboxed Unpickler."""
    text = path_instruction["instructions"]
    assert "pickle" in text.lower()
    assert "pickle.load()" in text or "pickle.Unpickler" in text


def test_instructions_requires_restricted_unpickler(path_instruction):
    """Constraint: a custom restricted Unpickler must be used for .pt files."""
    text = path_instruction["instructions"]
    assert "Unpickler" in text or "restricted" in text.lower()


def test_instructions_applies_pickling_constraint_to_pt_files(path_instruction):
    """Constraint must explicitly scope pickling rules to PyTorch .pt files."""
    text = path_instruction["instructions"]
    assert ".pt" in text


# ---------------------------------------------------------------------------
# Architectural constraint #5 — Flat Memory Profile
# ---------------------------------------------------------------------------

def test_instructions_prohibits_loading_tensor_weights(path_instruction):
    """Constraint: tensor weight matrices must never be loaded into memory."""
    text = path_instruction["instructions"]
    assert "Flat Memory" in text or "memory" in text.lower()


def test_instructions_requires_header_only_reads(path_instruction):
    """Constraint: code must stop after reading the metadata header."""
    text = path_instruction["instructions"]
    assert "header" in text.lower() or "metadata" in text.lower()


def test_instructions_contains_all_five_constraints(path_instruction):
    """Regression guard: all five numbered constraints must be present."""
    text = path_instruction["instructions"]
    for i in range(1, 6):
        assert f"{i}." in text, f"Constraint #{i} not found in instructions"


# ---------------------------------------------------------------------------
# chat section
# ---------------------------------------------------------------------------

def test_chat_is_mapping(config):
    assert isinstance(config["chat"], dict)


def test_chat_auto_reply_is_true(config):
    """auto_reply must be enabled so CodeRabbit responds to follow-up comments."""
    assert config["chat"]["auto_reply"] is True


# ---------------------------------------------------------------------------
# Boundary / regression tests
# ---------------------------------------------------------------------------

def test_config_has_no_unexpected_top_level_keys(config):
    """Guard against accidental extra keys that could indicate a misconfiguration."""
    known_keys = {"language", "reviews", "chat"}
    extra = set(config.keys()) - known_keys
    assert not extra, f"Unexpected top-level keys in .coderabbit.yaml: {extra}"


def test_reviews_section_has_expected_keys(config):
    """All expected reviews keys must be present (guards against typos/renames)."""
    expected = {
        "profile",
        "request_changes_workflow",
        "high_level_summary",
        "poem",
        "review_status",
        "collapse_walkthrough",
        "path_instructions",
    }
    missing = expected - set(config["reviews"].keys())
    assert not missing, f"Missing reviews keys: {missing}"


def test_instructions_text_is_substantial(path_instruction):
    """The instruction text must be long enough to be meaningful (>200 chars)."""
    assert len(path_instruction["instructions"]) > 200


def test_yaml_file_is_utf8_encodable():
    """The file must be readable as UTF-8 without errors (no binary garbage)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert len(content) > 0


def test_reviews_boolean_fields_are_actual_booleans(config):
    """Ensure YAML parsed the boolean flags as Python bools, not strings."""
    boolean_fields = [
        "request_changes_workflow",
        "high_level_summary",
        "poem",
        "review_status",
        "collapse_walkthrough",
    ]
    reviews = config["reviews"]
    for field in boolean_fields:
        assert isinstance(reviews[field], bool), (
            f"reviews.{field} should be a bool, got {type(reviews[field]).__name__}"
        )


def test_chat_auto_reply_is_actual_boolean(config):
    assert isinstance(config["chat"]["auto_reply"], bool)
