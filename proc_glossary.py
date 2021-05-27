
import spacy
import scispacy

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from textacy.ke import textrank


def is_nu_like(s):
    """
    Check if the given string looks like a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_spacy_nlp( model_name="en_core_sci_lg"):
    """
    Get a spacy model with a modified tokenizer which keeps words with hyphens together.
        For example, x-ray will not be split into "x" and "ray".

    Args:
        model_name: Name of spacy model to modify tokenizer for

    Returns:
        nlp: modified spacy model which does not split tokens on hyphens
    """
    nlp = spacy.load(model_name)

    # modify tokenizer infix patterns to not split on hyphen
    # Need to get words like x-ray and gamma-ray
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # EDIT: commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp


def process_keyword (doc):
    """
    Extract keywords from a single spacy doc using SingleRank

    Args:
        doc: Spacy doc from which to extract keywords

    Returns:
        text: The lemmatized and lowercase text for the doc
        kwd_counts: The SingleRank keywords with their scores and counts.
    """
    # SingleRank parameters
    kwds = textrank(
        doc,
        normalize="lemma",
        topn=999_999,  # This could cause issues with a huge abstract
        window_size=10,
        edge_weighting="count",
        position_bias=False,
    )
    # Remove keywords which are 1 character long or are numbers
    kwds = [(k.lower(), v) for k, v in kwds if (not is_nu_like(k)) and (len(k) > 1)]
    text = " ".join([t.lemma_.lower() for t in doc])
    kwd_counts = [(k, v) for k, v in kwds]
    # TODO: "Black hole nos with xnosnosx" would count "nos" 3 times. Do we want this?
    # If make match " nos ", need to keep in mind problems at beginning and end
    # of a sentence, and around punctuation, parentheses, etc.
    return kwd_counts


def load_terms(filename:str)->list:

    result = []
    with open(filename, 'r') as pg:
        data = pg.readlines()
        for d in data:
            result.append(d.strip())

    return result

nlp = get_spacy_nlp()

pterms = []
for t in load_terms('planetary_glossary.txt.raw'):
#for t in load_terms('helio_glossary.txt.raw'):
    doc = nlp(t)
    pterms.append([v[0] for v in process_keyword(doc)])

with open ('planetary_glossary.txt', 'w') as pout:
#with open ('helio_glossary.txt', 'w') as pout:
    for pt in pterms:
        term = f"%s" % " ".join(pt).strip()
        if term != "":
            pout.write(f"%s\n" % term)


