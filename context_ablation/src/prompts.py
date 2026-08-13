#!/usr/bin/env python
"""Prompt construction for the five context modes and three templates.

Design rule: **M1 and M2 must reproduce the benchmark's own prompts verbatim.** Those two modes are
what makes the study comparable to the main table -- the verification step checks that A1 at M2
reproduces the published `gpt-oss-120b_unblind` balanced accuracy of 0.554 -- so their text is
copied from `models/openrouter_LLM_{blind,unblind}/src/model_cfg.yml` and
`models/c2s_scale_2B_unblind/src/predict.py` rather than paraphrased. The benchmark's blind and
unblind templates differ by more than the candidate list (rule 2 changes too), so both are kept.

Modes M3-M5 extend from there:
  M3/M4 insert a tissue-and-species sentence.
  M5 drops the cell sentence entirely -- the control that separates reading the expression data
        from exploiting the label prior for the tissue.

Per-arm asymmetry, verified in code and NOT to be tidied away: the C2S template already names the
organism, so for that arm M3/M4 add only tissue, while for the instruction template they add
species and tissue. The modes are therefore not identical interventions across arms.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Benchmark templates, copied verbatim. Do not reword: M1/M2 are the comparability anchor.
# --------------------------------------------------------------------------- #
INSTRUCTION_NO_TYPES = """You are an expert at classifying cell types based on gene expression.
Given below is a list of up to {num_genes} gene names ordered by descending expression level in a single cell.
Your task is to identify the exact cell type this gene profile{context} represents.
Cell sentence: {cell_sentence}
Important rules:
1. Respond with only the cell type label of the predicted cell type.
2. Use the canonical cell type label from the OBO Foundry Cell Ontology (CL) — the primary label string as it appears in CL, not a synonym, abbreviation, or free-form description.
3. Do not include explanations, reasoning, or extra words in the output.
4. Do not repeat the question or input in the output.
Output:
"""

INSTRUCTION_WITH_TYPES = """You are an expert at classifying cell types based on gene expression.
Given below is a list of up to {num_genes} gene names ordered by descending expression level in a single cell.
Your task is to identify the exact cell type this gene profile{context} represents from this list of cell types: {unique_cell_types}.
Cell sentence: {cell_sentence}
Important rules:
1. Respond with only the cell type label of the predicted cell type.
2. Do only choose a cell type from the list of cell types provided.
3. Do not include explanations, reasoning, or extra words in the output.
4. Do not repeat the question or input in the output.
Output:
"""

# M5: same instructions, no gene list. The sentence naming the input is dropped rather than left
# empty, so the model is not invited to hallucinate a missing input.
INSTRUCTION_NO_GENES = """You are an expert at classifying cell types based on gene expression.
Your task is to name the most likely cell type{context} from this list of cell types: {unique_cell_types}.
No gene expression information is provided for this cell.
Important rules:
1. Respond with only the cell type label of the predicted cell type.
2. Do only choose a cell type from the list of cell types provided.
3. Do not include explanations, reasoning, or extra words in the output.
4. Do not repeat the question or input in the output.
Output:
"""

# C2S-Scale's own template, from models/c2s_scale_2B_unblind/src/predict.py. {organism} is already
# present here, which is the asymmetry noted above.
C2S_WITH_TYPES = ("The following is a list of {num_genes} gene names ordered by descending "
                  "expression level in a {organism}{context} cell. Your task is to give the cell "
                  "type which this cell belongs to based on its gene expression. Pick from this "
                  "list: {unique_cell_types}.\nCell sentence: {cell_sentence}.\n"
                  "The cell type corresponding to these genes is:")

C2S_NO_TYPES = ("The following is a list of {num_genes} gene names ordered by descending "
                "expression level in a {organism}{context} cell. Your task is to give the cell "
                "type which this cell belongs to based on its gene expression.\n"
                "Cell sentence: {cell_sentence}.\n"
                "The cell type corresponding to these genes is:")

C2S_NO_GENES = ("Your task is to give the cell type of a {organism}{context} cell. Pick from this "
                "list: {unique_cell_types}.\nNo gene expression information is provided for "
                "this cell.\nThe cell type corresponding to this cell is:")

# A4: one prompt per cluster, from its differentially expressed genes. Mirrors the established
# cluster-annotation phrasing (marker genes in, one label out) so the arm is comparable to the
# cluster-level baselines rather than to a bespoke prompt.
MARKER_WITH_TYPES = """You are an expert at annotating cell clusters from single-cell RNA sequencing data.
Given below are the top {num_genes} marker genes of one cell cluster, ordered by how strongly they distinguish it from all other clusters.
Your task is to identify the cell type of this{context} cluster from this list of cell types: {unique_cell_types}.
Marker genes: {marker_genes}
Important rules:
1. Respond with only the cell type label of the predicted cell type.
2. Do only choose a cell type from the list of cell types provided.
3. Do not include explanations, reasoning, or extra words in the output.
4. Do not repeat the question or input in the output.
Output:
"""

MARKER_NO_TYPES = """You are an expert at annotating cell clusters from single-cell RNA sequencing data.
Given below are the top {num_genes} marker genes of one cell cluster, ordered by how strongly they distinguish it from all other clusters.
Your task is to identify the cell type of this{context} cluster.
Marker genes: {marker_genes}
Important rules:
1. Respond with only the cell type label of the predicted cell type.
2. Use the canonical cell type label from the OBO Foundry Cell Ontology (CL) — the primary label string as it appears in CL, not a synonym, abbreviation, or free-form description.
3. Do not include explanations, reasoning, or extra words in the output.
4. Do not repeat the question or input in the output.
Output:
"""


def candidate_list(labels, rng):
    """The candidate cell types for ONE cell, independently shuffled.

    The benchmark permutes the list per cell -- ``predict.py:477`` builds
    ``[rng.permutation(labels).tolist() for _ in range(adata.n_obs)]`` from a seeded generator --
    so every cell sees a different order. That is deliberate: LLMs have position bias, and a fixed
    order would let a model score well by favouring whatever sits first. Reproducing it is required
    both for comparability and for the study to be measuring what it claims.

    Kept as an explicit call rather than hidden inside ``build_prompt`` so a caller cannot forget
    it: passing an unshuffled list is a silent bias, not an error.

    ``.tolist()``, NOT ``list()``, and the difference is not cosmetic. ``list()`` on a numpy array
    yields ``np.str_`` scalars, and under numpy 2.x their repr is ``np.str_('B cell')`` rather than
    ``'B cell'``. Since the instruction template renders the candidates by repr, ``list()`` puts
    ``[np.str_('retinal progenitor cell'), np.str_('retinal rod cell'), ...]`` into the prompt --
    81 characters of numpy noise for a 9-type dataset, and a list that reads as code rather than as
    cell types. ``.tolist()`` converts to plain Python strings, which is what the benchmark sends.
    Getting this wrong silently breaks the M2 anchor that the whole study is calibrated against.
    """
    return rng.permutation(list(labels)).tolist()


def render_candidates(template, labels):
    """Render the candidate list the way each template's own code path does.

    The two differ, and the difference is visible in the prompt text:

    * ``instruction`` -- the benchmark passes the LIST OBJECT into ``str.format``, so it renders as
      a Python repr with brackets and quotes: ``['neutrophil', 'regulatory t cell']``. Confirmed
      against the worked example in ``scLLMs_benchmark_paper/main.tex``.
    * ``c2s`` / ``marker_genes`` -- ``models/c2s_scale_2B_unblind/src/predict.py`` builds
      ``", ".join(cell_type_list)``, so it renders bare.

    Reproducing each faithfully is what makes M1/M2 comparable to the published numbers.
    """
    labels = list(labels)
    return repr(labels) if template == "instruction" else ", ".join(labels)


def short_tissue(tissue):
    """A tissue phrase short enough to read inline.

    The dataset YAMLs carry tissue strings of wildly different length, because they were written to
    give blind mode a usable hint rather than to be embedded in a sentence:

        human_pancreas   'pancreas'                                    8 chars
        HFRA             'retina (fetal)'                             14
        HIHA             'blood (peripheral blood immune cells)'      37
        Zeisel           'nervous system (whole mouse CNS and PNS:    159
                          cortex, hippocampus, ... enteric ganglia)'

    Inserted verbatim, the tissue context would be a whole clause for Zeisel and one word for
    pancreas, so "what naming the tissue buys" would not be comparable across datasets and Zeisel
    would likely show the largest gain for reasons unrelated to the model. Truncating at the first
    parenthesis or colon keeps a head noun that still reads naturally in the sentence:

        'pancreas', 'retina', 'blood', 'nervous system'

    Override per dataset in config.yml (`context_phrase`) when the head noun is not the natural
    term -- 'brain' rather than 'nervous system', say.
    """
    if not tissue:
        return ""
    head = str(tissue).split("(")[0].split(":")[0]
    return " ".join(head.split()).strip(" ,;")


def context_phrase(mode, template, tissue=None, species=None, override=None):
    """Inline species-and-tissue phrase, e.g. ``" mouse brain"``, or ``""`` when withheld.

    Always returns either the empty string or a string with exactly ONE leading space, so the
    templates interpolate it directly inside a sentence and collapse cleanly to the benchmark's
    wording when the mode withholds context. That property is what keeps M1 and M2 byte-identical
    to the published prompts, and ``_self_test`` asserts it rather than trusting it.

    The C2S template already names the organism, so species is omitted there to avoid stating it
    twice ("a Homo sapiens mouse cell").
    """
    if not mode.get("tissue_species"):
        return ""
    if override:
        return " " + " ".join(str(override).split())
    parts = []
    if species and template != "c2s":
        parts.append(str(species).strip())
    tis = short_tissue(tissue)
    if tis:
        parts.append(tis)
    phrase = " ".join(p for p in parts if p)
    return f" {phrase}" if phrase else ""


def build_prompt(template, mode, *, num_genes=None, cell_sentence=None, unique_cell_types=None,
                 marker_genes=None, organism=None, tissue=None, species=None,
                 context_override=None):
    """One prompt for one (template, mode).

    ``mode`` is the dict from config.yml: ``{cell_types, tissue_species, genes}``.
    """
    ctx = context_phrase(mode, template, tissue=tissue, species=species,
                         override=context_override)
    # Per-template rendering, and the caller is responsible for having shuffled the list already
    # (see candidate_list). Joining here would silently diverge from the benchmark's prompts.
    types_str = render_candidates(template, unique_cell_types) if unique_cell_types else ""

    if template == "marker_genes":
        if not mode.get("genes"):
            raise ValueError("the cluster arm has no meaning without its marker genes: a cluster "
                             "prompt is nothing but the gene list, so mode M5 does not apply")
        body = MARKER_WITH_TYPES if mode.get("cell_types") else MARKER_NO_TYPES
        return body.format(num_genes=num_genes, marker_genes=", ".join(marker_genes),
                           unique_cell_types=types_str, context=ctx)

    if template == "c2s":
        if not mode.get("genes"):
            return C2S_NO_GENES.format(organism=organism, unique_cell_types=types_str, context=ctx)
        body = C2S_WITH_TYPES if mode.get("cell_types") else C2S_NO_TYPES
        return body.format(num_genes=num_genes, organism=organism, cell_sentence=cell_sentence,
                           unique_cell_types=types_str, context=ctx)

    if template == "instruction":
        if not mode.get("genes"):
            return INSTRUCTION_NO_GENES.format(unique_cell_types=types_str, context=ctx)
        body = INSTRUCTION_WITH_TYPES if mode.get("cell_types") else INSTRUCTION_NO_TYPES
        return body.format(num_genes=num_genes, cell_sentence=cell_sentence,
                           unique_cell_types=types_str, context=ctx)

    raise ValueError(f"unknown template: {template}")


def _self_test():
    """Check the property the comparability of the whole study rests on.

    M1 and M2 under the instruction template must equal the benchmark's blind and unblind prompts
    exactly, including the absence of any stray whitespace where the context clause would go.
    """
    m1 = {"cell_types": False, "tissue_species": False, "genes": True}
    m2 = {"cell_types": True, "tissue_species": False, "genes": True}
    m4 = {"cell_types": True, "tissue_species": True, "genes": True}

    p1 = build_prompt("instruction", m1, num_genes=200, cell_sentence="A B C")
    assert "represents.\nCell sentence:" in p1, "M1 gained stray whitespace before the sentence"
    assert "list of cell types" not in p1, "M1 must not name candidate types"

    p2 = build_prompt("instruction", m2, num_genes=200, cell_sentence="A B C",
                      unique_cell_types=["b cell", "t cell"])
    # The instruction template renders the list as a Python repr, brackets and quotes included,
    # because the benchmark passes the list object into str.format. Not a comma-joined string.
    assert "from this list of cell types: ['b cell', 't cell'].\nCell sentence:" in p2, p2
    assert "2. Do only choose a cell type from the list" in p2, "M2 must use the unblind rule 2"

    # C2S joins bare, so the two templates must NOT render the list the same way.
    c2 = build_prompt("c2s", m2, num_genes=200, cell_sentence="A B C", organism="Homo sapiens",
                      unique_cell_types=["b cell", "t cell"])
    assert "Pick from this list: b cell, t cell." in c2, c2

    # Shuffling is per cell and seeded; two cells must be able to differ.
    import numpy as np
    rng = np.random.default_rng(0)
    labels = [f"type {i}" for i in range(12)]
    draws = {tuple(candidate_list(labels, rng)) for _ in range(8)}
    assert len(draws) > 1, "candidate_list must shuffle per cell, not return a fixed order"

    # Context reads inline, and the long tissue strings are truncated to a head noun that fits.
    p4 = build_prompt("instruction", m4, num_genes=200, cell_sentence="A B C",
                      unique_cell_types=["b cell"],
                      tissue="nervous system (whole mouse CNS and PNS: cortex, hippocampus)",
                      species="mouse")
    assert "this gene profile mouse nervous system represents from this list" in p4, p4

    # The C2S template already names the organism, so species must not be repeated there.
    c4 = build_prompt("c2s", m4, num_genes=200, cell_sentence="A B C", organism="Homo sapiens",
                      unique_cell_types=["b cell"], tissue="blood (peripheral blood immune cells)",
                      species="human")
    assert "in a Homo sapiens blood cell." in c4, c4
    assert "human" not in c4.split("Cell sentence")[0].replace("Homo sapiens", ""), \
        "C2S already names the organism; species must not be duplicated"

    # Per-dataset override, for when the head noun is not the natural term.
    p4b = build_prompt("instruction", m4, num_genes=200, cell_sentence="A B C",
                       unique_cell_types=["b cell"], tissue="nervous system (...)",
                       species="mouse", context_override="mouse brain")
    assert "this gene profile mouse brain represents" in p4b, p4b

    # --- whitespace: no double spaces, and no space before punctuation, in ANY combination ------
    import itertools
    tissues = [None, "", "pancreas", "retina (fetal)",
               "nervous system (whole mouse CNS and PNS: cortex, hippocampus)"]
    for tmpl, mode, tis, spec in itertools.product(
            ("instruction", "c2s"),
            ({"cell_types": ct, "tissue_species": ts, "genes": g}
             for ct in (True, False) for ts in (True, False) for g in (True, False)),
            tissues, (None, "mouse")):
        text = build_prompt(tmpl, mode, num_genes=200, cell_sentence="A B C",
                            organism="Homo sapiens", unique_cell_types=["b cell"],
                            tissue=tis, species=spec)
        assert "  " not in text, f"double space in {tmpl} {mode} tissue={tis!r}:\n{text}"
        for bad in (" .", " ,", " :", " ;"):
            assert bad not in text, f"space before punctuation in {tmpl} {mode}:\n{text}"

    # M5 has no meaning for the cluster arm.
    m5 = {"cell_types": True, "tissue_species": True, "genes": False}
    try:
        build_prompt("marker_genes", m5, num_genes=25, marker_genes=["A"])
    except ValueError:
        pass
    else:
        raise AssertionError("the cluster arm should reject the no-genes mode")

    print("prompts self-test: OK")


if __name__ == "__main__":
    _self_test()
