#!/usr/bin/env python3
"""Read SBS96 contexts directly from annotated VCF files.

This module intentionally does not query a reference genome.  It is used when
the VCF already contains the trinucleotide sequence (or a complete SBS96
label) in an INFO field, which makes VCF processing independent of species and
genome build.
"""

import gzip
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import pandas as pd


BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGT", "TGCA")
DEFAULT_CONTEXT_TAGS = ("TRINUCLEOTIDE", "TRINUC", "TRI", "CONTEXT", "SBS96")
SUBSTITUTIONS = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
SBS96_CONTEXTS = tuple(
    "{}[{}]{}".format(left, substitution, right)
    for left in BASES
    for substitution in SUBSTITUTIONS
    for right in BASES
)


class ContextVCFError(ValueError):
    """Raised when a context-annotated VCF cannot be interpreted safely."""


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def _complement(base):
    return base.translate(COMPLEMENT)


def _canonical_label(trinucleotide, reference, alternate):
    if len(trinucleotide) != 3 or any(base not in BASES for base in trinucleotide):
        raise ContextVCFError(
            "Trinucleotide context must contain exactly three A/C/G/T bases: {}".format(
                trinucleotide
            )
        )
    if trinucleotide[1] != reference:
        raise ContextVCFError(
            "The middle base of context {} does not match reference allele {}.".format(
                trinucleotide, reference
            )
        )
    if reference == alternate:
        raise ContextVCFError("Reference and alternate alleles must be different.")

    if reference in "AG":
        trinucleotide = reverse_complement(trinucleotide)
        reference = _complement(reference)
        alternate = _complement(alternate)

    label = "{}[{}>{}]{}".format(
        trinucleotide[0], reference, alternate, trinucleotide[2]
    )
    if label not in SBS96_CONTEXTS:
        raise ContextVCFError(
            "Context does not describe an SBS96 mutation: {}".format(label)
        )
    return label


def _alleles_match_vcf(context_ref, context_alt, vcf_ref, vcf_alt):
    return (context_ref, context_alt) in (
        (vcf_ref, vcf_alt),
        (_complement(vcf_ref), _complement(vcf_alt)),
    )


def normalize_sbs96_context(value, reference, alternate):
    """Normalize an annotated context to the pyrimidine-oriented SBS96 label.

    Supported values include a reference trinucleotide (``ACA``), an SBS96
    label (``A[C>A]A``), and paired reference/alternate trinucleotides
    (``ACA>AAA`` or ``ACA/AAA``).
    """

    reference = str(reference).strip().upper()
    alternate = str(alternate).strip().upper()
    value = str(value).strip().strip('"').strip("'").upper().replace(" ", "")

    if reference not in BASES or alternate not in BASES:
        raise ContextVCFError(
            "Only single-nucleotide A/C/G/T alleles can be assigned to SBS96."
        )

    match = re.fullmatch(r"([ACGT])\[([ACGT])>([ACGT])\]([ACGT])", value)
    if match:
        left, context_ref, context_alt, right = match.groups()
        if not _alleles_match_vcf(context_ref, context_alt, reference, alternate):
            raise ContextVCFError(
                "Annotated substitution {}>{} does not match VCF alleles {}>{}.".format(
                    context_ref, context_alt, reference, alternate
                )
            )
        return _canonical_label(left + context_ref + right, context_ref, context_alt)

    match = re.fullmatch(r"([ACGT]{3})[>/]([ACGT]{3})", value)
    if match:
        reference_context, alternate_context = match.groups()
        if (
            reference_context[0] != alternate_context[0]
            or reference_context[2] != alternate_context[2]
        ):
            raise ContextVCFError(
                "Reference and alternate trinucleotides must have identical flanking bases."
            )
        context_ref = reference_context[1]
        context_alt = alternate_context[1]
        if not _alleles_match_vcf(context_ref, context_alt, reference, alternate):
            raise ContextVCFError(
                "Annotated substitution {}>{} does not match VCF alleles {}>{}.".format(
                    context_ref, context_alt, reference, alternate
                )
            )
        return _canonical_label(reference_context, context_ref, context_alt)

    if re.fullmatch(r"[ACGT]{3}", value):
        context_ref = value[1]
        if context_ref == reference:
            context_alt = alternate
        elif context_ref == _complement(reference):
            # Also accept a raw trinucleotide that has already been oriented to
            # the pyrimidine strand while REF/ALT remain in genomic orientation.
            context_alt = _complement(alternate)
        else:
            raise ContextVCFError(
                "The middle base of context {} does not match VCF REF {} on either strand.".format(
                    value, reference
                )
            )
        return _canonical_label(value, context_ref, context_alt)

    raise ContextVCFError(
        "Unsupported context value {!r}; expected ACA, A[C>A]A, ACA>AAA, or ACA/AAA.".format(
            value
        )
    )


def _vcf_paths(samples):
    path = Path(samples)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ContextVCFError("VCF input does not exist: {}".format(samples))

    paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name.lower().endswith(".vcf")
            or candidate.name.lower().endswith(".vcf.gz")
            or candidate.name.lower().endswith(".vcf.bgz")
        )
    )
    if not paths:
        raise ContextVCFError(
            "No .vcf, .vcf.gz, or .vcf.bgz files were found in {}.".format(samples)
        )
    return paths


def _open_vcf(path):
    if path.name.lower().endswith((".gz", ".bgz")):
        return gzip.open(str(path), "rt")
    return open(str(path), "rt")


def _filename_sample(path):
    name = path.name
    for suffix in (".vcf.bgz", ".vcf.gz", ".vcf"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _parse_info(info_text):
    result = {}
    if info_text in ("", "."):
        return result
    for item in info_text.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = True
    return result


def _find_context_tag(info, requested_tag):
    tags_by_uppercase = {tag.upper(): tag for tag in info}
    if requested_tag and requested_tag.upper() != "AUTO":
        return tags_by_uppercase.get(requested_tag.upper())
    for tag in DEFAULT_CONTEXT_TAGS:
        if tag in tags_by_uppercase:
            return tags_by_uppercase[tag]
    return None


def _value_for_alt(raw_value, alt_index, alt_count):
    values = str(raw_value).split(",")
    if len(values) == 1:
        return values[0]
    if len(values) == alt_count:
        return values[alt_index - 1]
    raise ContextVCFError(
        (
            "Context INFO field has {} values for {} ALT alleles; "
            "expected one value or Number=A."
        ).format(len(values), alt_count)
    )


def _samples_with_alt(format_text, sample_values, sample_names, alt_index):
    if not sample_names:
        return []
    format_keys = format_text.split(":") if format_text not in ("", ".") else []
    if "GT" not in format_keys:
        return list(sample_names)

    gt_index = format_keys.index("GT")
    selected = []
    for sample_name, sample_value in zip(sample_names, sample_values):
        fields = sample_value.split(":")
        if gt_index >= len(fields):
            continue
        genotype = fields[gt_index]
        alleles = re.split(r"[/|]", genotype)
        if str(alt_index) in alleles:
            selected.append(sample_name)
    return selected


def read_context_annotated_vcfs(samples, context_tag="AUTO"):
    """Return an SBS96 count matrix and per-mutation table from VCF input."""

    sample_order = OrderedDict()
    mutations = []
    skipped_non_snv = 0

    for path in _vcf_paths(samples):
        fallback_sample = _filename_sample(path)
        header_samples = []
        saw_header = False

        with _open_vcf(path) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line or line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    columns = line.split("\t")
                    if len(columns) < 8:
                        raise ContextVCFError(
                            "{}:{}: malformed #CHROM header.".format(path, line_number)
                        )
                    header_samples = columns[9:]
                    if header_samples:
                        for sample_name in header_samples:
                            sample_order.setdefault(sample_name, None)
                    else:
                        sample_order.setdefault(fallback_sample, None)
                    saw_header = True
                    continue
                if line.startswith("#"):
                    continue
                if not saw_header:
                    raise ContextVCFError(
                        "{}:{}: VCF data encountered before #CHROM header.".format(
                            path, line_number
                        )
                    )

                fields = line.split("\t")
                if len(fields) < 8:
                    raise ContextVCFError(
                        "{}:{}: expected at least 8 tab-delimited VCF columns.".format(
                            path, line_number
                        )
                    )
                if header_samples and len(fields) < 9 + len(header_samples):
                    raise ContextVCFError(
                        "{}:{}: expected values for {} VCF sample column(s).".format(
                            path, line_number, len(header_samples)
                        )
                    )

                chromosome, position, record_id, reference, alternate_text = fields[:5]
                try:
                    position = int(position)
                except ValueError as error:
                    raise ContextVCFError(
                        "{}:{}: VCF POS must be an integer, got {!r}.".format(
                            path, line_number, position
                        )
                    ) from error
                reference = reference.upper()
                alternates = [
                    alternate.upper() for alternate in alternate_text.split(",")
                ]
                info = _parse_info(fields[7])
                tag = _find_context_tag(info, context_tag)

                for alt_index, alternate in enumerate(alternates, start=1):
                    if (
                        len(reference) != 1
                        or len(alternate) != 1
                        or reference not in BASES
                        or alternate not in BASES
                    ):
                        skipped_non_snv += 1
                        continue
                    if tag is None:
                        requested = context_tag or "AUTO"
                        raise ContextVCFError(
                            "{}:{}: SBS record is missing INFO/{} trinucleotide context.".format(
                                path, line_number, requested
                            )
                        )

                    try:
                        context_value = _value_for_alt(
                            info[tag], alt_index, len(alternates)
                        )
                        mutation_type = normalize_sbs96_context(
                            context_value, reference, alternate
                        )
                    except ContextVCFError as error:
                        raise ContextVCFError(
                            "{}:{}: {}".format(path, line_number, error)
                        ) from error

                    if header_samples:
                        format_text = fields[8] if len(fields) > 8 else "."
                        sample_values = fields[9:]
                        record_samples = _samples_with_alt(
                            format_text,
                            sample_values,
                            header_samples,
                            alt_index,
                        )
                    else:
                        record_samples = [fallback_sample]

                    for sample_name in record_samples:
                        sample_order.setdefault(sample_name, None)
                        mutations.append(
                            {
                                "Sample Names": sample_name,
                                "Chr": str(chromosome),
                                "Pos": position,
                                "ID": record_id,
                                "Ref": reference,
                                "Alt": alternate,
                                "MutationType": mutation_type,
                            }
                        )

    if skipped_non_snv:
        warnings.warn(
            "Skipped {} non-SNV ALT allele(s); trinucleotide input supports SBS96 only.".format(
                skipped_non_snv
            ),
            UserWarning,
        )
    if not mutations:
        raise ContextVCFError(
            "No sample SNVs with usable trinucleotide contexts were found."
        )

    sample_names = list(sample_order.keys())
    matrix = pd.DataFrame(0, index=SBS96_CONTEXTS, columns=sample_names, dtype=int)
    for mutation in mutations:
        matrix.at[mutation["MutationType"], mutation["Sample Names"]] += 1
    matrix.index.name = "MutationType"

    mutation_table = pd.DataFrame(
        mutations,
        columns=[
            "Sample Names",
            "Chr",
            "Pos",
            "ID",
            "Ref",
            "Alt",
            "MutationType",
        ],
    )
    return matrix, mutation_table
