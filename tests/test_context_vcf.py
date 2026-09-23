import pandas as pd
import pytest

from SigProfilerAssignment.context_vcf import (
    ContextVCFError,
    normalize_sbs96_context,
    read_context_annotated_vcfs,
)
from SigProfilerAssignment.decompose_subroutines import probabilities_per_mutation


@pytest.mark.parametrize(
    "context,ref,alt,expected",
    [
        ("ACA", "C", "A", "A[C>A]A"),
        ("TGT", "G", "T", "A[C>A]A"),
        ("ACA", "G", "T", "A[C>A]A"),
        ("A[C>A]A", "C", "A", "A[C>A]A"),
        ("A[C>A]A", "G", "T", "A[C>A]A"),
        ("ACA>AAA", "C", "A", "A[C>A]A"),
    ],
)
def test_normalize_sbs96_context(context, ref, alt, expected):
    assert normalize_sbs96_context(context, ref, alt) == expected


def test_normalize_rejects_context_that_does_not_match_ref():
    with pytest.raises(ContextVCFError, match="does not match VCF REF"):
        normalize_sbs96_context("ATA", "C", "A")


def test_read_context_annotated_multisample_vcf(tmp_path):
    vcf_path = tmp_path / "non_model_species.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##INFO=<ID=TRINUCLEOTIDE,Number=1,Type=String,Description=\"Reference trinucleotide\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tfish_A\tfish_B\n"
        "scaffold_1\t10\trs1\tC\tA\t.\tPASS\tTRINUCLEOTIDE=ACA\tGT\t0/1\t0/0\n"
        "scaffold_2\t20\t.\tG\tT\t.\tPASS\tTRINUCLEOTIDE=TGT\tGT\t0/0\t1/1\n"
    )

    matrix, mutations = read_context_annotated_vcfs(
        str(vcf_path), context_tag="TRINUCLEOTIDE"
    )

    assert matrix.shape == (96, 2)
    assert matrix.at["A[C>A]A", "fish_A"] == 1
    assert matrix.at["A[C>A]A", "fish_B"] == 1
    assert mutations["Sample Names"].tolist() == ["fish_A", "fish_B"]
    assert mutations["Chr"].tolist() == ["scaffold_1", "scaffold_2"]
    assert mutations["MutationType"].tolist() == ["A[C>A]A", "A[C>A]A"]


def test_read_sites_only_vcf_uses_filename_as_sample_and_auto_tag(tmp_path):
    vcf_path = tmp_path / "sample_x.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "contigA\t7\t.\tT\tC\t.\tPASS\tContext=A[T>C]G\n"
    )

    matrix, mutations = read_context_annotated_vcfs(str(vcf_path), context_tag="AUTO")

    assert list(matrix.columns) == ["sample_x"]
    assert matrix.at["A[T>C]G", "sample_x"] == 1
    assert mutations.loc[0, "Sample Names"] == "sample_x"


def test_multiallelic_vcf_uses_number_a_contexts_and_genotype(tmp_path):
    vcf_path = tmp_path / "multiallelic.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_A\n"
        "contigA\t9\t.\tC\tA,G\t.\tPASS\tTRI=ACA,ACA\tGT\t1/2\n"
    )

    matrix, mutations = read_context_annotated_vcfs(str(vcf_path), context_tag="AUTO")

    assert matrix.at["A[C>A]A", "sample_A"] == 1
    assert matrix.at["A[C>G]A", "sample_A"] == 1
    assert mutations["Alt"].tolist() == ["A", "G"]


def test_missing_context_tag_has_file_and_line_in_error(tmp_path):
    vcf_path = tmp_path / "missing.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t3\t.\tC\tA\t.\tPASS\t.\n"
    )

    with pytest.raises(ContextVCFError, match=r"missing\.vcf:3:.*TRINUCLEOTIDE"):
        read_context_annotated_vcfs(
            str(vcf_path), context_tag="TRINUCLEOTIDE"
        )


def test_probabilities_per_mutation_accepts_in_memory_records():
    probability_matrix = pd.DataFrame(
        {
            "MutationType": ["A[C>A]A"],
            "SBS1": [0.25],
            "SBS5": [0.75],
        },
        index=pd.Index(["fish_A"], name="Sample Names"),
    )
    mutation_records = pd.DataFrame(
        [
            {
                "Sample Names": "fish_A",
                "Chr": "scaffold_1",
                "Pos": 10,
                "ID": "rs1",
                "Ref": "C",
                "Alt": "A",
                "MutationType": "A[C>A]A",
            }
        ]
    )

    matrices, samples = probabilities_per_mutation(
        probability_matrix,
        samples_path="path-that-must-not-be-read",
        m="96",
        mutation_records=mutation_records,
    )

    assert samples == ["fish_A"]
    assert matrices[0].loc[0, "SBS1"] == 0.25
    assert matrices[0].loc[0, "SBS5"] == 0.75
    assert matrices[0].loc[0, "ID"] == "rs1"
