import logging
import random
from unittest import TestCase

import numpy as np
import pandas as pd

import src.create_keyword_and_syn_lists as ks

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Test(TestCase):

    def setUp(self) -> None:
        self.min_thresh = 0
        records = [
            {"arxiv_class": ["cond-mat.stat-mech", "cond-mat.dis-nn", "cs.NI",
                             "math-ph", "nlin.AO", "physics.data-an"],
             "alternate_bibcode": ["2001cond.mat..6096A"],
             "bibcode": "2002RvMP...74...47A",
             "keyword": ["05.20.-y", "89.20.Hh", "05.40.-a", "01.30.Vv", "02.10.-v",
                         "02.40.Pc", "02.50.-r", "82.20.Wt",
                         "Nonlinear Sciences - Adaptation and Self-Organizing Systems",
                         "Physics - Data Analysis", "Statistics and Probability"],
             "database": ["physics", "general"],
             "abstract": "Complex networks describe a wide range of systems in nature and society. Frequently cited examples include the cell, a network of chemicals linked by chemical reactions, and the Internet, a network of routers and computers connected by physical links. While traditionally these systems have been modeled as random graphs, it is increasingly recognized that the topology and evolution of real networks are governed by robust organizing principles. This article reviews the recent advances in the field of complex networks, focusing on the statistical mechanics of network topology and dynamics. After reviewing the empirical data that motivated the recent interest in networks, the authors discuss the main models and analytical tools, covering random graphs, small-world and scale-free networks, the emerging theory of evolving networks, and the interplay between topology and the network's robustness against failures and attacks. <P \/>",
             "bibstem": ["RvMP", "RvMP...74"], "citation_count": 6196,
             "title": "Statistical mechanics of complex networks", "year": 2002,
             "rake_kwds": [["frequently cited examples include", 16.0],
                           ["robust organizing principles", 9.0],
                           ["scale-free networks", 8.1428571429],
                           ["motivated", 1.0], ["interplay", 1.0], ["failures", 1.0],
                           ["attacks", 1.0], ["<p \/>", 1.0]]},
            {"arxiv_class": ["cond-mat.mtrl-sci"],
             "alternate_bibcode": ["2001cond.mat..4182S"],
             "bibcode": "2002JPCM...14.2745S",
             "keyword": ["Condensed Matter - Materials Science"],
             "database": ["physics", "general"],
             "abstract": "We have developed and implemented a selfconsistent density functional method using standard norm-conserving pseudopotentials and a flexible, numerical linear combination of atomic orbitals basis set, which includes multiple-zeta and polarization orbitals. Exchange and correlation are treated with the local spin density or generalized gradient approximations. The basis functions and the electron density are projected on a real-space grid, in order to calculate the Hartree and exchange-correlation potentials and matrix elements, with a number of operations that scales linearly with the size of the system. We use a modified energy functional, whose minimization produces orthogonal wavefunctions and the same energy and density as the Kohn-Sham energy functional, without the need for an explicit orthogonalization. Additionally, using localized Wannier-like electron wavefunctions allows the computation time and memory required to minimize the energy to also scale linearly with the size of the system. Forces and stresses are also calculated efficiently and accurately, thus allowing structural relaxation and molecular dynamics simulations. <P \/>",
             "bibstem": ["JPCM", "JPCM...14"], "citation_count": 4823,
             "title": "The SIESTA method for ab initio order-N materials simulation",
             "year": np.nan,
             "rake_kwds": [["ab initio order-n materials simulation", 33.5],
                           ["localized wannier-like electron wavefunctions allows",
                            33.0],
                           ["minimization produces orthogonal wavefunctions", 17.0],
                           ["standard norm-conserving pseudopotentials", 16.0],
                           ["additionally", 1.0], ["minimize", 1.0], ["forces", 1.0],
                           ["stresses", 1.0], ["accurately", 1.0], ["<p \/>", 1.0]]},
        ]
        yc = list(range(1996, 2009)) + [np.nan]
        synth_records = np.tile(records, 500).tolist()
        df = pd.DataFrame(synth_records)
        df['year'] = df['year'].apply(lambda x: random.choice(yc))
        df['database'] = df['database'].apply(lambda x: x + ['astronomy'])
        df.iloc[[10, 20], :].loc[:, ['year']] = np.nan
        self.df = df

    def test_flatten_to_keywords(self):
        self.kwd_df, self.year_counts = ks.flatten_to_keywords(self.df, self.min_thresh)
        self.assertNotEqual(self.kwd_df.shape[0], 0)
        self.assertNotEqual(len(self.year_counts), 0)
