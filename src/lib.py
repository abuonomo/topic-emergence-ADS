# Functions for determing whether a paper from the ADS has NASA affiliations
# author: Brian Thomas (brian.a.thomas@nasa.gov)

import json
import re

NASA_grant_pattern = re.compile(r"NASA grant", re.IGNORECASE)


def is_nasa_pub(doc: dict) -> bool:
    # NASA afiliation list (from the ADS, see https://github.com/csgrant00/CanonicalAffiliations/blob/master/parent_child.tsv)
    NASA_Afiliations = [
        "A00676",
        "A00948",
        "A00677",
        "A00678",
        "A00680",
        "A00681",
        "A00682",
        "A00684",
        "A00685",
        "A00686",
        "A00687",
        "A00688",
        "A11075",
        "A03987",
        "A01099",
    ]
    SpaceTel = "A03671"
    IPAC = "A01097"
    # other afilliated NASA funded institutions?

    # get the afiliation list and see if ANY of the listed afilliations matches a NASA institution
    if "afil_id" in doc:
        for afil in doc["aff_id"]:
            if afil in NASA_Afiliations:
                return True

    return False


def is_nasa_grant_acknowledgement(doc: dict) -> bool:
    # get the afiliation list and see if ANY of the listed afilliations matches a NASA institution
    if "ack" in doc and NASA_grant_pattern.search(doc["ack"]):
        return True

    return False


def get_paper_nasa_status(doc: dict) -> str:

    status = "NO"

    # get if nasa author
    is_NASA_pub = is_nasa_pub(doc)

    if is_NASA_pub:
        status = "YES"  # NASA_afiliation"
    elif is_nasa_grant_acknowledgement(doc):
        # check for acknowledgement as a grant
        status = "YES"  # NASA_grant"

    return status


def get_papers_nasa_status(docs: list) -> dict:
    # go thru docs and try to determine number of NASA publications
    status = dict()
    for doc in docs:
        doc_id = doc["bibcode"]
        status[doc_id] = get_paper_nasa_status(doc)

    return status


def contains_element(L1: list, L2: list) -> bool:
    test = [i for i in L1 if i in L2]
    if len(test) > 0:
        return True
    return False


def gather_data(data_dir: str, min_chars: int = 99) -> dict:

    papers = dict()

    # for each file, load and process our data

    from os import listdir
    from os.path import isfile, join

    data_files = [
        join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))
    ]

    allowed_database = ["astronomy"]
    # load our data
    for data_file in data_files:

        print(f"Doing %s" % data_file)
        with open(data_file, encoding="utf-8-sig") as f:
            data = json.load(f)

        # filtered_docs = [d for d in data['docs'] if contains_element(d['database'],allowed_database)]
        filtered_docs = [
            d
            for d in data["docs"]
            if contains_element(d["database"], allowed_database)
            and "astronomy" in d["database"]
            and "abstract" in d
            and len(d["abstract"]) > min_chars
        ]
        print(f"  got %s matches for filtered list" % len(filtered_docs))

        # init our year of data
        year = data["year"]
        papers[year] = get_papers_nasa_status(filtered_docs)

    return papers
