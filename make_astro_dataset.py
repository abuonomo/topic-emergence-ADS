
""" Short program to build out a dataset of astro-only publication document abstracts """
import concurrent.futures
import json
import glob
import logging
import os

import spacy
import scispacy

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from textacy.ke import textrank

LOG = logging.getLogger('makeastoronly')
logging.basicConfig(level=logging.ERROR)

num_threads = 15 

def load_terms(filename:str)->list:

    result = []
    with open(filename, 'r') as pg:
        data = pg.readlines()
        for d in data:
            result.append(d.strip())

    return result

HELIO_TERMS = load_terms('helio_glossary.txt')
PLANETARY_TERMS = load_terms('planetary_glossary.txt')
PROHIBITED_TERMS = set(HELIO_TERMS + PLANETARY_TERMS)

# Viz 17
#ALLOWED_BIBS = ['AJ', 'AJS', 'ApJ', 'ApJS', 'MNRAS', 'A&A', 'A&ARv', 'A&AS', 'PASP']

# Viz 18
# Attempt at a larger set of journals
# all bibs not dedicated to physics/astronomy or general science (ex Nature, PLOS One, Proc National Academy, Science, PhD Thesis, etc) 
# We dropped journals which focus on:
# electronics, math, planetary, biology, geosci, environmental or solar 1998-2010 
# computer science, engineering, quantum theory,
# knocked out ArXiv, GCN, yCat (Vizier) & MPC/MPEC (Minor planet catalogs) too 
# Further filter: need more than 5000 papers
#ALLOWED_BIBS = ['A&A', 'AAS', 'ACP', 'ACPD', 'AIAAJ', 'AIPC', 'AJ', 'AMM', 'APS', 'ASAJ', 'ASPC', 'ASSL', 'ATel', 'AcAau', 'AcPPB', 'AcSpA', 'AdSpR', 'AmJPh', 'AmMin', 'Ana', 'Ap&SS', 'ApJ', 'ApJL', 'ApOpt', 'ApPhA', 'ApPhB', 'ApPhL', 'ApSS', 'ApSpe', 'CBET', 'CMaPh', 'CP', 'CPL', 'ChPhB', 'ChPhL', 'ChPhy', 'ChSBu', 'CoPhC', 'CoTPh', 'DPS', 'EL', 'EPJA', 'EPJB', 'EPJC', 'EPJD', 'ESASP', 'HyInt', 'IAUC', 'IAUS', 'ICRC', 'IEDL', 'IEITC', 'IEITF', 'IJBC', 'IJMPA', 'IJMPB', 'IJMSp', 'IJNME', 'IJRS', 'IJSSC', 'IJTIA', 'IJTPE', 'IPTL', 'ITAP', 'ITAS', 'ITED', 'ITEIS', 'ITGRS', 'ITIP', 'ITM', 'ITMTT', 'ITNS', 'ITSP', 'JAP', 'JAerS', 'JApSc', 'JChPh', 'JCoPh', 'JETP', 'JETPL', 'JHEP', 'JKPS', 'JMP', 'JPSJ', 'JPhA', 'JPhB', 'JPhCS', 'JPhD', 'JPhG', 'JPhy4', 'JQSRT', 'JTePh', 'JaJAP', 'LNP', 'MNRAS', 'MPLA', 'MPLB', 'MmSAI', 'NJPh', 'NYASA', 'Natur', 'NewSc', 'PLoSO', 'PNAS', 'PhDT', 'PhLA', 'PhLB', 'PhRvA', 'PhRvB', 'PhRvC', 'PhRvD', 'PhRvE', 'PhRvL', 'PhST', 'PhT', 'RSPTA', 'S&T', 'SPIE', 'STIN', 'Sci', 'SciAm', 'TePhL', 'cosp', 'cxo', 'hst', 'yCat']

# Viz 19
# These are the journals with high impact, as gauged from SJR index 
#Unlike T-R index which compares number of citations to average of 2 yrs number of articles, 
# SJR (maddeningly) does a running 3yr average. So not apples-to-apples, but ‘close’.
#I think we probably want everything with a value of ‘1’ or higher (so I clipped the list there).
'''
['Annual Review of Astronomy and Astrophysics:14.12266666666667',
 'Astrophysical Journal, Supplement Series:6.856833333333332',
 'Astrophysical Journal Letters:5.4195',
NO- 'Annual Review of Earth and Planetary Sciences:5.1330833333333326',
 'Astronomy and Astrophysics Review:4.489916666666667',
 'Astronomical Journal:4.130666666666666',
 'Monthly Notices of the Royal Astronomical Society:3.4601666666666664',
 'Astrophysical Journal:3.219416666666667',
NO-'Icarus:2.683',
 'Astronomy and Astrophysics:2.6445',
 'Publications of the Astronomical Society of the Pacific:2.422083333333333',
 'Astroparticle Physics:2.0084166666666663',
 'Acta Astronomica:1.956583333333333',
NO- 'Physics of the Earth and Planetary Interiors:1.9470833333333335',
 'Publication of the Astronomical Society of Japan:1.8160833333333333',
NO- 'Space Science Reviews:1.647166666666667',
 'New Astronomy:1.5964999999999998',
NO- 'Solar Physics:1.5615833333333333',
NO- 'Annales Geophysicae:1.2300833333333334',
NO- 'Planetary and Space Science:1.0425000000000002',
NO- 'Living Reviews in Solar Physics:1.0228333333333335',
 'Publications of the Astronomical Society of Australia:1.0019999999999998',
 ]
'''
#ALLOWED_BIBS = ['ARA&A', 'ApJS', 'ApJL', 'A&ARv', 'AJ', 'MNRAS', 'ApJ', 'A&A', 'PASP', 'APh', 'AcAau', 'PASJ', 'NewA', 'PASA' ]

# Viz 20
# as for Viz19, but the high impact journals for 2007-2019 period of time.
'''
['Annual Review of Astronomy and Astrophysics:18.545846153846153',
 'Astrophysical Journal, Supplement Series:7.05523076923077',
NO- 'Annual Review of Earth and Planetary Sciences:5.941000000000001',
 'Astronomy and Astrophysics Review:5.664076923076923',
NO- 'Living Reviews in Solar Physics:5.062307692307693',
 'Astrophysical Journal Letters:3.2809999999999993',
 'Astrophysical Journal:3.1380769230769228',
 'Astronomical Journal:3.111307692307693',
 'Monthly Notices of the Royal Astronomical Society:2.943615384615384',
 'Astronomy and Astrophysics:2.662153846153846',
NO- 'Space Science Reviews:2.633',
 'Publications of the Astronomical Society of the Pacific:2.393384615384616',
NO- 'Icarus:2.336461538461538',
 'New Astronomy Reviews:2.0703846153846155',
 'Astroparticle Physics:1.9091538461538466',
NO- 'Physics of the Earth and Planetary Interiors:1.893846153846154',
 'Physics of the Dark Universe:1.8833076923076921',
 'Monthly Notices of the Royal Astronomical Society: Letters:1.8094615384615382',
 'Acta Astronomica:1.636923076923077',
NO- 'Solar Physics:1.629923076923077',
 'Publications of the Astronomical Society of Australia:1.601076923076923',
 'Publication of the Astronomical Society of Japan:1.5110000000000001',
 'Journal of Cosmology and Astroparticle Physics:1.1855384615384619',
NO- 'Planetary and Space Science:1.1573846153846152',
NO- 'Annales Geophysicae:1.1452307692307693',
 ]

'''
ALLOWED_BIBS = ['ARA&A', 'ApJS', 'ApJL', 'A&ARv', 'ApJ', 'AJ', 'MNRAS', 'A&A', 'PASP', 'NewAR', 'APh', 'PDU', 'AcAau', 'PASA', 'PASJ', 'JCAP']

INPUT_DATA_DIR = 'data/raw_2020'
OUTPUT_DATA_DIR = 'data/astro_only_2020'

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


def process_keyword (doc)->str:
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
    #text = " ".join([t.lemma_.lower() for t in doc])
    return " ".join([k for k, v in kwds])

def has_no_prohibited_terms(terms:list)->bool:
    for t in terms:
        if t in PROHIBITED_TERMS:
            return False
    return True

nlp = get_spacy_nlp()

def make_astro_only_dataset(f:str)->dict:

    bibstems = {}
    LOG.error(f"Reading {f}")

    collection = {"year": 0, "numFound": 0, "docs":[]}
    with open(f, 'r') as json_file:
        data = json.load(json_file)

        # record year
        collection['year'] = data['year']

        # build out/filter acceptable documents from basic list
        for d in data['docs']:

            # must be from acceptable journal 
            for b in d['bibstem']:
                if b in ALLOWED_BIBS:
                    # this is something we will use
                    # 1. Must have an abstract and of suitable length
                    #2. Must have keywords and NO prohibited keywords
                    if 'abstract' in d and len(d['abstract']) > 100 and \
                       'keyword' in d and has_no_prohibited_terms([ process_keyword(nlp(k)) for k in d['keyword']]):

                        # IF we get here, then add it to the collection 
                        collection['docs'].append(d)

                        # get some stats
                        if b not in bibstems:
                            bibstems[b] = 1
                        else:
                            bibstems[b] += 1

                    # there may be more than one bibstem, but we record only 1x 
                    break 

    # record size
    collection['numFound'] = len(collection['docs'])

    # write out the collection for this year
    #
    filename = os.path.join(OUTPUT_DATA_DIR,os.path.basename(f))
    LOG.error(f"Writing {filename}") 
    with open(filename, 'w') as out:
        json.dump(collection, out)

    return bibstems

# snag a list of ADS Json files which contain the abstracts
# we wish to extract
files = glob.glob(f"{INPUT_DATA_DIR}/*.json.new")

print (files)

# thread on files and process
with concurrent.futures.ThreadPoolExecutor(max_workers = num_threads) as executor:

    future_to_proc = { executor.submit(make_astro_only_dataset, f): f for f in files}
    for future in concurrent.futures.as_completed(future_to_proc):
        proc = future_to_proc[future]
        try:
            bibstems = future.result()
            # some stats on journals
            LOG.error({k: v for k, v in sorted(bibstems.items(), key=lambda item: item[1], reverse=True)})
        except Exception as exc:
            LOG.fatal('FAILED to run exception: %s' % ( exc))

