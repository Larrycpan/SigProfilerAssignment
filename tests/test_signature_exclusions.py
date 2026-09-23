import pytest

from SigProfilerAssignment.decomposition import resolve_signature_exclusions
from SigProfilerAssignment.decompose_subroutines import (
    qualify_signature_exclusions,
    validate_explicit_signature_exclusions,
)


def test_none_resolves_to_empty_lists():
    assert resolve_signature_exclusions(None, "SBS") == ([], [])


def test_mixed_subgroup_and_explicit_signature():
    resolved, explicit = resolve_signature_exclusions(
        ["MMR_deficiency_signatures", "SBS42"], "SBS"
    )

    assert resolved == [
        "SBS6",
        "SBS14",
        "SBS15",
        "SBS20",
        "SBS21",
        "SBS26",
        "SBS44",
        "SBS42",
    ]
    assert explicit == ["SBS42"]


def test_duplicate_and_case_normalization():
    resolved, explicit = resolve_signature_exclusions(
        ["MMR_deficiency_signatures", "sbs6", "sbs10A", "SBS10a"], "SBS"
    )

    assert resolved.count("SBS6") == 1
    assert resolved.count("SBS10a") == 1
    assert explicit == ["SBS6", "SBS10a"]


@pytest.mark.parametrize(
    "exclusions, mutation_prefix, error_type",
    [
        ("SBS42", "SBS", TypeError),
        (["unknown_group"], "SBS", ValueError),
        (["DBS1"], "SBS", ValueError),
        ([42], "SBS", TypeError),
        (["SBS42"], "CNV", ValueError),
    ],
)
def test_invalid_exclusions_raise(exclusions, mutation_prefix, error_type):
    with pytest.raises(error_type):
        resolve_signature_exclusions(exclusions, mutation_prefix)


def test_legacy_bare_ids_are_still_qualified():
    assert qualify_signature_exclusions(
        ["42", "SBS6", "sbs10a", "42"], "SBS"
    ) == ["SBS42", "SBS6", "SBS10a"]


def test_explicit_signature_must_exist_in_database():
    with pytest.raises(ValueError, match="SBS42"):
        validate_explicit_signature_exclusions(
            ["SBS42"], ["SBS1", "SBS5", "SBS6"]
        )


def test_explicit_signature_validation_accepts_available_signature():
    validate_explicit_signature_exclusions(["SBS42"], ["SBS1", "SBS42"])
