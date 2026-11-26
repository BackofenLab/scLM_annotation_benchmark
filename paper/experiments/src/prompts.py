#!/usr/bin/env python
"""Prompt templates and prompt construction for the context modes.

Three templates: ``instruction``, the prompt of the benchmark's OpenRouter wrapper; ``c2s``, the
prompt of the C2S-Scale wrapper; ``marker_genes``, one prompt per cluster. A mode (``modes`` in
config.yml) sets whether a prompt lists the candidate cell types, names tissue and species, and
holds the genes. Without tissue and species, the instruction and c2s prompts equal the wrappers'
prompts without the list (M1) and with it (M2). The c2s template names the organism already, so
tissue and species add only the tissue there.

    python src/prompts.py    # self-test
"""
from __future__ import annotations

# The prompts of the benchmark's OpenRouter wrapper (models/openrouter_LLM_blind and
# models/openrouter_LLM_unblind), with a slot for the context phrase.
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

# For a mode without genes (genes: false): the sentence that introduces the genes is left out.
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

# The prompts of the C2S-Scale wrapper (models/c2s_scale_2B_blind and models/c2s_scale_2B_unblind),
# with a slot for the context phrase.
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

# For a mode without genes.
C2S_NO_GENES = ("Your task is to give the cell type of a {organism}{context} cell. Pick from this "
                "list: {unique_cell_types}.\nNo gene expression information is provided for "
                "this cell.\nThe cell type corresponding to this cell is:")

# One prompt per cluster, from its marker genes, with the rules of the instruction template.
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

#: The option to abstain, offered by the imperfect-list conditions (``lists`` in config.yml).
UNCERTAIN_RULE = "If none of the listed cell types fits, answer uncertain."


def _offer_uncertain(body, mode):
    """``body`` with the option to answer "uncertain" if ``mode`` offers it, else unchanged.

    The instruction and marker templates get it as rule 5, the C2S template right after its list.
    """
    if not mode.get("uncertain"):
        return body
    if not mode.get("cell_types"):
        raise ValueError("'uncertain' is an alternative to the listed cell types; it needs a list")
    if body.endswith("Output:\n"):
        return body.removesuffix("Output:\n") + f"5. {UNCERTAIN_RULE}\nOutput:\n"
    return body.replace("{unique_cell_types}.", "{unique_cell_types}. " + UNCERTAIN_RULE)


def candidate_list(labels, rng):
    """The candidate cell types for one cell, in an order drawn from ``rng``.

    The benchmark's OpenRouter wrapper shuffles the list per cell in the same way. ``.tolist()``
    returns plain strings; numpy string scalars would appear as ``np.str_('...')`` in the prompt.
    """
    return rng.permutation(list(labels)).tolist()


def ordered_candidates(template, labels, rng):
    """The candidate cell types for one cell, in the order of the template's benchmark wrapper.

    The C2S-Scale wrapper gives every cell the sorted list, so ``c2s`` gets it sorted without a
    draw from ``rng``; the other templates get a list shuffled per cell (``candidate_list``).
    """
    return sorted(labels) if template == "c2s" else candidate_list(labels, rng)


def render_candidates(template, labels):
    """The candidate list as prompt text.

    ``instruction`` gets the repr of the list, ``['neutrophil', 'regulatory t cell']``, because
    the OpenRouter wrapper formats the list object into its template. ``c2s`` and ``marker_genes``
    get the names joined by ``", "``, as in the C2S-Scale wrapper.
    """
    labels = list(labels)
    return repr(labels) if template == "instruction" else ", ".join(labels)


def short_tissue(tissue):
    """The tissue up to its first parenthesis or colon: 'retina (fetal)' becomes 'retina'.

    The tissues in the dataset YAMLs range from 8 to 159 characters; the head noun keeps the
    inserted phrase about equally short for every dataset.
    """
    if not tissue:
        return ""
    head = str(tissue).split("(")[0].split(":")[0]
    return " ".join(head.split()).strip(" ,;")


def context_phrase(mode, template, tissue=None, species=None, override=None):
    """Species and tissue as an inline phrase with one leading space, e.g. ``" mouse brain"``.

    Returns ``""`` when the mode gives no tissue and species, which leaves the benchmark's wording.
    ``override`` (``context_phrase`` in a dataset YAML) replaces the phrase. The species is left
    out for the C2S template, which names the organism already.
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
    """One prompt for one template and mode.

    ``mode`` is an entry of ``modes`` in config.yml or a list condition (``lists.condition_modes``).
    """
    ctx = context_phrase(mode, template, tissue=tissue, species=species,
                         override=context_override)
    # The caller orders the list (ordered_candidates).
    types_str = render_candidates(template, unique_cell_types) if unique_cell_types else ""

    if template == "marker_genes":
        if not mode.get("genes"):
            raise ValueError("the cluster arm has no meaning without its marker genes: a cluster "
                             "prompt is nothing but the gene list, so mode M5 does not apply")
        body = MARKER_WITH_TYPES if mode.get("cell_types") else MARKER_NO_TYPES
        return _offer_uncertain(body, mode).format(
            num_genes=num_genes, marker_genes=", ".join(marker_genes),
                           unique_cell_types=types_str, context=ctx)

    if template == "c2s":
        if not mode.get("genes"):
            return _offer_uncertain(C2S_NO_GENES, mode).format(
                organism=organism, unique_cell_types=types_str, context=ctx)
        body = C2S_WITH_TYPES if mode.get("cell_types") else C2S_NO_TYPES
        return _offer_uncertain(body, mode).format(
            num_genes=num_genes, organism=organism, cell_sentence=cell_sentence,
            unique_cell_types=types_str, context=ctx)

    if template == "instruction":
        if not mode.get("genes"):
            return _offer_uncertain(INSTRUCTION_NO_GENES, mode).format(
                unique_cell_types=types_str, context=ctx)
        body = INSTRUCTION_WITH_TYPES if mode.get("cell_types") else INSTRUCTION_NO_TYPES
        return _offer_uncertain(body, mode).format(
            num_genes=num_genes, cell_sentence=cell_sentence, unique_cell_types=types_str,
            context=ctx)

    raise ValueError(f"unknown template: {template}")


def _self_test():
    """Checks of the wording of M1 and M2, the list, the context phrase, the whitespace and the
    "uncertain" option."""
    m1 = {"cell_types": False, "tissue_species": False, "genes": True}
    m2 = {"cell_types": True, "tissue_species": False, "genes": True}
    m4 = {"cell_types": True, "tissue_species": True, "genes": True}

    p1 = build_prompt("instruction", m1, num_genes=200, cell_sentence="A B C")
    assert "represents.\nCell sentence:" in p1, "M1 gained stray whitespace before the sentence"
    assert "list of cell types" not in p1, "M1 must not name candidate types"

    p2 = build_prompt("instruction", m2, num_genes=200, cell_sentence="A B C",
                      unique_cell_types=["b cell", "t cell"])
    # The instruction template writes the list as a Python repr.
    assert "from this list of cell types: ['b cell', 't cell'].\nCell sentence:" in p2, p2
    assert "2. Do only choose a cell type from the list" in p2, "M2 must use the unblind rule 2"

    # The C2S template joins the names with commas.
    c2 = build_prompt("c2s", m2, num_genes=200, cell_sentence="A B C", organism="Homo sapiens",
                      unique_cell_types=["b cell", "t cell"])
    assert "Pick from this list: b cell, t cell." in c2, c2

    # Two cells can get different orders.
    import numpy as np
    rng = np.random.default_rng(0)
    labels = [f"type {i}" for i in range(12)]
    draws = {tuple(candidate_list(labels, rng)) for _ in range(8)}
    assert len(draws) > 1, "candidate_list must shuffle per cell, not return a fixed order"

    # The context phrase is inline, with the tissue cut to its head noun.
    p4 = build_prompt("instruction", m4, num_genes=200, cell_sentence="A B C",
                      unique_cell_types=["b cell"],
                      tissue="nervous system (whole mouse CNS and PNS: cortex, hippocampus)",
                      species="mouse")
    assert "this gene profile mouse nervous system represents from this list" in p4, p4

    # No species in the C2S template, which names the organism.
    c4 = build_prompt("c2s", m4, num_genes=200, cell_sentence="A B C", organism="Homo sapiens",
                      unique_cell_types=["b cell"], tissue="blood (peripheral blood immune cells)",
                      species="human")
    assert "in a Homo sapiens blood cell." in c4, c4
    assert "human" not in c4.split("Cell sentence")[0].replace("Homo sapiens", ""), \
        "C2S already names the organism; species must not be duplicated"

    # A dataset's override replaces the phrase.
    p4b = build_prompt("instruction", m4, num_genes=200, cell_sentence="A B C",
                       unique_cell_types=["b cell"], tissue="nervous system (...)",
                       species="mouse", context_override="mouse brain")
    assert "this gene profile mouse brain represents" in p4b, p4b

    # "uncertain" adds its rule and nothing else: taking the rule out gives back the M2 prompt.
    mu = {**m2, "uncertain": True}
    for tmpl, extra in (("instruction", {}), ("c2s", {"organism": "Homo sapiens"}),
                        ("marker_genes", {"marker_genes": ["A", "B"]})):
        kw = dict(num_genes=200, cell_sentence="A B C", unique_cell_types=["b cell"], **extra)
        offered = build_prompt(tmpl, mu, **kw)
        assert offered.count(UNCERTAIN_RULE) == 1, offered
        stripped = offered.replace(f"5. {UNCERTAIN_RULE}\n", "").replace(f" {UNCERTAIN_RULE}", "")
        assert stripped == build_prompt(tmpl, m2, **kw), offered
    try:
        build_prompt("instruction", {**m1, "uncertain": True}, num_genes=200, cell_sentence="A")
    except ValueError:
        pass
    else:
        raise AssertionError("'uncertain' without a list must be refused")

    # No double spaces and no space before punctuation, in any combination.
    import itertools
    tissues = [None, "", "pancreas", "retina (fetal)",
               "nervous system (whole mouse CNS and PNS: cortex, hippocampus)"]
    for tmpl, mode, tis, spec in itertools.product(
            ("instruction", "c2s"),
            ({"cell_types": ct, "tissue_species": ts, "genes": g, "uncertain": u}
             for ct in (True, False) for ts in (True, False) for g in (True, False)
             for u in ((False, True) if ct else (False,))),
            tissues, (None, "mouse")):
        text = build_prompt(tmpl, mode, num_genes=200, cell_sentence="A B C",
                            organism="Homo sapiens", unique_cell_types=["b cell"],
                            tissue=tis, species=spec)
        assert "  " not in text, f"double space in {tmpl} {mode} tissue={tis!r}:\n{text}"
        for bad in (" .", " ,", " :", " ;"):
            assert bad not in text, f"space before punctuation in {tmpl} {mode}:\n{text}"

    # The marker template refuses a mode without genes.
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
