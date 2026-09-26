"""Query transformation modes -- the independent variable of the study.

Every mode has the same contract:

    mode(query) -> (text_to_send_to_remote_model, meta_dict)

`transformer(query, mode)` keeps the original entry point from pproxy.py.
Adding H1-H4 later means writing one function and adding one MODES entry;
nothing in the runner needs to change.
"""

import llm_clients

# What "rephrasing" means in this project, spelled out for the local model.
# It is NOT synonym substitution -- it is raising the level of abstraction so
# that the remote model can still answer usefully while never seeing who the
# user is, where they work, or what happened to them. Small local models
# drift into commentary ("Sure! Here is a rephrased...") when the instruction
# is loose, and that preamble would then be sent onward as if it were the
# query, so the output rule is stated bluntly at the end.
REPHRASE_SYSTEM = """You are a privacy-preserving rewriter. You sit between a \
user and a powerful cloud model that must NOT be trusted with personal details.

Rephrasing here does not mean swapping words for synonyms. It means restating \
the user's request at a higher level of abstraction:

1. Remove every identifying or sensitive detail - names, employers, schools, \
cities, ages, amounts, dates, diagnoses, relationships. If a detail is \
needed for the answer to make sense, generalise it instead of deleting it \
("WIPRO, Noida" becomes "a company"; "my 7 year old son" becomes "a young \
child").
2. Keep the information need completely intact. The cloud model must still be \
able to produce an answer that actually helps the original user.
3. Make the request self-contained. The cloud model never sees the original \
query, so nothing may be left implicit.
4. Write it as a generic, third-person request. Do not roleplay as the user \
and do not invent details that were not there.

Output ONLY the rewritten request. Do not answer it, do not explain what you \
changed, do not add quotes, labels or preamble."""


# One worked example, shown to the model as a demonstration. Note what it
# does: the employer and city are gone, the first-person account becomes a
# task description, and the actual need (a formal application email that
# states qualifications and asks for consideration) survives untouched.
EXAMPLE_ORIGINAL = """dear HR i have applied for the post of processor in \
operations at WIPRO, Noida. I have the knowledge and skills required for this \
job post. i have attached my resume, i hope you will consider my application \
for this post"""

EXAMPLE_REPHRASED = """Please generate a formal email to the Human Resources \
department of a company, expressing interest in a job opening for a processor \
in the operations department. The email should include a statement of \
qualifications and a polite expression of hope for consideration."""

REPHRASE_PROMPT = (
    "Here is an example of the rewriting you must perform.\n\n"
    "Original query:\n" + EXAMPLE_ORIGINAL + "\n\n"
    "Rewritten query:\n" + EXAMPLE_REPHRASED + "\n\n"
    "Now rewrite the query below in exactly the same way.\n\n"
    "Original query:\n{query}\n\n"
    "Rewritten query:\n"
)


def baseline(query):
    """H0: send the query through untouched."""
    return query, {"transform": "baseline"}


def rephrase(query):
    """Local model rephrases; the rephrasing is what gets sent onward."""
    prompt = REPHRASE_PROMPT.format(query=query)
    text, meta = llm_clients.ollama_generate(prompt, system=REPHRASE_SYSTEM)
    text = _strip_wrapping(text)
    return text, {"transform": "rephrase", "rephraser": meta}


def _strip_wrapping(text):
    """Remove quotes/prefixes a small local model tends to add anyway."""
    text = text.strip()
    for prefix in ("Rewritten query:", "Rewritten question:",
                   "Original query:", "Question:", "Rewritten:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


MODES = {
    "baseline": baseline,
    "rephrase": rephrase,
    # Spelling in the original stub was "repharse"; kept as an alias so old
    # commands keep working.
    "repharse": rephrase,
}

# Modes run when --modes is not given. The aliases above are deliberately
# excluded so a default run does not do the same work twice.
DEFAULT_MODES = ("baseline", "rephrase")


def transformer(query, mode):
    """Apply `mode` to `query`. Returns (transformed_query, meta)."""
    try:
        fn = MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(MODES)}"
        ) from None
    return fn(query)

# SUGGESTION: the rephraser is the one place the raw query still exists in
# full, so it is worth logging its output even on failure -- if a rephrasing
# silently drops a PII span (say the name in q025), that is a finding for
# H1/H2, not a bug to discard.
#
# SUGGESTION: for the H3 chunking arm, keep the chunk boundaries in meta.
# Re-synthesis quality is hard to interpret later without knowing where the
# query was cut.
