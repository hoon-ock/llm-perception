"""
Single pipeline: extends fc_group/functional_group_dataset.csv in place with
new functional-group molecules (halogen family, O/S/N parallels, position
isomers), sourcing their property columns as reliably as possible, and
overwrites the file with the combined result.

Sources, by column:
    mw, tpsa, hbd, hba, avg_carbon_oxidation_state
        -> computed offline and exactly from SMILES via RDKit.
    boiling_point_c, water_solubility
        -> fetched from PubChem PUG View's "Experimental Properties" section
           (best-effort; not every compound has data there -- left NaN and
           flagged rather than guessed).
    pka, pkah
        -> no reliable free structured API exists for these. Filled in from
           standard literature pKa values (MANUAL_PKA below), same as how
           the original dataset's pKa/pKaH columns were hand-curated -- one
           representative value per functional-group class rather than a
           per-molecule lookup.

Only the molecule *identity* (name, SMILES, functional group, chain length)
is authored directly below -- that defines which molecule it is, not a
property to look up.
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

ORIGINAL_CSV = 'fc_group/functional_group_dataset.csv'
OUTPUT_CSV = 'fc_group/functional_group_dataset.csv'
COLUMNS = [
    'iupac_name', 'common_name', 'formula', 'smiles', 'carbon_count', 'functional_group',
    'functional_group_structure', 'mw', 'pka', 'pkah', 'tpsa',
    'avg_carbon_oxidation_state', 'hbd', 'hba', 'boiling_point_c', 'water_solubility',
]

PUBCHEM_CID_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON'
PUBCHEM_VIEW_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading={heading}'
REQUEST_DELAY_S = 0.25
ELECTRONEGATIVE = {'O', 'N', 'S', 'F', 'Cl', 'Br', 'I'}

# One representative (pka, pkah) per new functional-group class, from standard
# literature pKa tables (e.g. Bordwell/Evans-style aqueous pKa compilations).
# NaN = no confident standard value to cite; left for future manual lookup
# rather than guessed.
MANUAL_PKA = {
    # alpha-C-H pKa carried over from the existing "alkyl halide" (Cl) convention;
    # halogen identity doesn't shift it enough to justify fabricated precision.
    'alkyl fluoride': (48, 'N/A (no conventional aqueous pKaH)'),
    'alkyl bromide':  (48, 'N/A (no conventional aqueous pKaH)'),
    'alkyl iodide':   (48, 'N/A (no conventional aqueous pKaH)'),
    # dialkyl ether/sulfide/sulfoxide conjugate-acid pKaH are well-documented;
    # their own pKa (no acidic proton of note) is left NaN.
    'ether':          (np.nan, -3.5),
    'thioether':      (np.nan, -6.8),
    'sulfoxide':      (np.nan, -1.8),
    'sulfone':        (np.nan, np.nan),  # no confident standard value for either
    # simple aldimine conjugate acid (iminium) pKaH ~7; imine N-H pKa not
    # a standard tabulated value.
    'imine':          (np.nan, 7.2),
    # nitroalkane alpha-C-H is a classic, well-documented acidic proton;
    # nitro basicity is negligible (no meaningful pKaH).
    'nitro':          (8.9, 'N/A (negligible basicity)'),
    # position isomers (propan-2-ol, propan-2-amine, propane-2-thiol, ...) are
    # merged into the primary alcohol/amine/thiol classes -- C1 vs C2 diff-vectors
    # measured at ~0.98-0.99 cosine similarity, so position doesn't distinguish
    # them. pKa/pKaH still differ slightly per exact molecule (isopropanol,
    # isopropylamine, isopropanethiol literature values), independent of the
    # shared class label.
    'alcohol':        (17.1, -2.35),
    'amine':          (35, 10.63),
    'thiol':          (10.9, -6.8),
}

SOLUBILITY_KEYWORDS = [
    ('miscible', 'miscible'),
    ('insoluble', 'very low'),
    ('very slightly soluble', 'very low'),
    ('slightly soluble', 'low'),
    ('sparingly soluble', 'low'),
    ('freely soluble', 'high'),
    ('very soluble', 'high'),
    ('soluble', 'moderate'),
]

# ==================================================
# Structure notation: one SMILES + one condensed form
# ==================================================
#
# The `formula` column mixes the two notations *and the notation is confounded
# with the class*: 9 of the 20 groups were written entirely as SMILES (the ones
# this script generates, which set formula=smiles) and 8 entirely as condensed
# structural formulas (the hand-written original rows). "Is this a SMILES
# string?" was therefore itself a usable feature -- for a probe reading the
# prompt, and for the character n-gram surface baseline alike. It is sharpest
# for the halides: F/Br/I are all SMILES, Cl is all condensed, which is exactly
# the split that made `alkyl chloride` the sole failing group under
# leave-one-group-out.
#
# `smiles` below is populated for *every* molecule from one set of templates, so
# that column carries no class information at all. `formula` is kept unchanged
# for backward compatibility with results already produced from it, but new
# prompts should use `smiles`.
#
# The condensed templates are NOT written to the CSV. They exist only as the
# cross-check in `verify_structures`: a condensed structural formula spells out
# every atom including hydrogens, so comparing it against RDKit's atom counts
# catches an off-by-one in a CH2 run that a "does the SMILES parse" check would
# sail past. `formula` already covers the condensed notation for any prompt that
# wants it -- though note it is the mixed, class-confounded column.
#
# Keyed by functional group, except alcohol/amine/thiol which contain both
# 1- and 2-substituted isomers and so need one template each. `n` is
# carbon_count. Verified in `verify_structures`: the SMILES must parse, and the
# condensed form must contain exactly the atoms RDKit finds in the SMILES.
SMILES_BY_STRUCTURE = {
    'alcohol_1':       lambda n: 'C' * n + 'O',
    'alcohol_2':       lambda n: 'CC(O)' + 'C' * (n - 2),
    'aldehyde':        lambda n: 'C' * (n - 1) + 'C=O',
    'alkyl bromide':   lambda n: 'C' * n + 'Br',
    'alkyl chloride':  lambda n: 'C' * n + 'Cl',
    'alkyl fluoride':  lambda n: 'C' * n + 'F',
    'alkyl iodide':    lambda n: 'C' * n + 'I',
    'amide':           lambda n: 'C' * (n - 1) + 'C(N)=O',
    'amine_1':         lambda n: 'C' * n + 'N',
    'amine_2':         lambda n: 'CC(N)' + 'C' * (n - 2),
    'carboxylic acid': lambda n: 'C' * (n - 1) + 'C(O)=O',
    'ester':           lambda n: 'C' * (n - 2) + 'C(=O)OC',
    'ether':           lambda n: 'C' * (n - 1) + 'OC',
    'imine':           lambda n: 'C' * (n - 1) + 'C=N',
    'ketone':          lambda n: 'CC(=O)' + 'C' * (n - 2),
    'nitrile':         lambda n: 'C' * (n - 1) + 'C#N',
    'nitro':           lambda n: 'C' * n + '[N+](=O)[O-]',
    'none (alkane)':   lambda n: 'C' * n,
    'sulfone':         lambda n: 'CS(=O)(=O)' + 'C' * (n - 1),
    'sulfoxide':       lambda n: 'CS(=O)' + 'C' * (n - 1),
    'thioether':       lambda n: 'C' * (n - 1) + 'SC',
    'thiol_1':         lambda n: 'C' * n + 'S',
    'thiol_2':         lambda n: 'CC(S)' + 'C' * (n - 2),
}

# Verification only -- see the note above. Written in one consistent style, fully
# explicit CH3/CH2 runs, so the parser below has a single form to handle.
CONDENSED_BY_STRUCTURE = {
    'alcohol_1':       lambda n: 'CH3' + 'CH2' * (n - 1) + 'OH',
    'alcohol_2':       lambda n: 'CH3CH(OH)' + 'CH2' * (n - 3) + 'CH3',
    'aldehyde':        lambda n: 'CH3' + 'CH2' * (n - 2) + 'CHO',
    'alkyl bromide':   lambda n: 'CH3' + 'CH2' * (n - 1) + 'Br',
    'alkyl chloride':  lambda n: 'CH3' + 'CH2' * (n - 1) + 'Cl',
    'alkyl fluoride':  lambda n: 'CH3' + 'CH2' * (n - 1) + 'F',
    'alkyl iodide':    lambda n: 'CH3' + 'CH2' * (n - 1) + 'I',
    'amide':           lambda n: 'CH3' + 'CH2' * (n - 2) + 'CONH2',
    'amine_1':         lambda n: 'CH3' + 'CH2' * (n - 1) + 'NH2',
    'amine_2':         lambda n: 'CH3CH(NH2)' + 'CH2' * (n - 3) + 'CH3',
    'carboxylic acid': lambda n: 'CH3' + 'CH2' * (n - 2) + 'COOH',
    'ester':           lambda n: 'CH3' + 'CH2' * (n - 3) + 'COOCH3',
    'ether':           lambda n: 'CH3' + 'CH2' * (n - 2) + 'OCH3',
    'imine':           lambda n: 'CH3' + 'CH2' * (n - 2) + 'CH=NH',
    'ketone':          lambda n: 'CH3CO' + 'CH2' * (n - 3) + 'CH3',
    'nitrile':         lambda n: 'CH3' + 'CH2' * (n - 2) + 'CN',
    'nitro':           lambda n: 'CH3' + 'CH2' * (n - 1) + 'NO2',
    'none (alkane)':   lambda n: 'CH3' + 'CH2' * (n - 2) + 'CH3',
    'sulfone':         lambda n: 'CH3SO2' + 'CH2' * (n - 2) + 'CH3',
    'sulfoxide':       lambda n: 'CH3SO' + 'CH2' * (n - 2) + 'CH3',
    'thioether':       lambda n: 'CH3' + 'CH2' * (n - 2) + 'SCH3',
    'thiol_1':         lambda n: 'CH3' + 'CH2' * (n - 1) + 'SH',
    'thiol_2':         lambda n: 'CH3CH(SH)' + 'CH2' * (n - 3) + 'CH3',
}

# Fragment -> atom counts, matched longest-first, for verifying the condensed
# strings against RDKit. Deliberately not a general condensed-formula parser --
# it only needs to cover the fragments the templates above emit, and an
# unrecognised fragment raises rather than being skipped.
_CONDENSED_FRAGMENTS = [
    ('COOCH3', dict(C=2, O=2, H=3)), ('CONH2', dict(C=1, O=1, N=1, H=2)),
    ('COOH', dict(C=1, O=2, H=1)), ('OCH3', dict(O=1, C=1, H=3)),
    ('SCH3', dict(S=1, C=1, H=3)), ('=NH', dict(N=1, H=1)),
    ('NH2', dict(N=1, H=2)), ('CHO', dict(C=1, H=1, O=1)),
    ('CH3', dict(C=1, H=3)), ('CH2', dict(C=1, H=2)), ('SO2', dict(S=1, O=2)),
    ('NO2', dict(N=1, O=2)), ('SO', dict(S=1, O=1)), ('OH', dict(O=1, H=1)),
    ('SH', dict(S=1, H=1)), ('CN', dict(C=1, N=1)), ('CO', dict(C=1, O=1)),
    ('CH', dict(C=1, H=1)), ('Cl', dict(Cl=1)), ('Br', dict(Br=1)),
    ('F', dict(F=1)), ('I', dict(I=1)),
]

# ============================
# New molecules (identity only)
# ============================

NEW_MOLECULES = [
    # --- Halogen family: fluoride, bromide, iodide (parallel to existing chloride rows) ---
    *[
        dict(iupac_name=f'1-Fluoro{stem}', common_name=f'n-{alkyl} fluoride', smiles=smiles,
             carbon_count=c, functional_group='alkyl fluoride', functional_group_structure='-F')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCF', 3), ('butane', 'butyl', 'CCCCF', 4),
            ('pentane', 'amyl', 'CCCCCF', 5), ('hexane', 'hexyl', 'CCCCCCF', 6),
        ]
    ],
    *[
        dict(iupac_name=f'1-Bromo{stem}', common_name=f'n-{alkyl} bromide', smiles=smiles,
             carbon_count=c, functional_group='alkyl bromide', functional_group_structure='-Br')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCBr', 3), ('butane', 'butyl', 'CCCCBr', 4),
            ('pentane', 'amyl', 'CCCCCBr', 5), ('hexane', 'hexyl', 'CCCCCCBr', 6),
        ]
    ],
    *[
        dict(iupac_name=f'1-Iodo{stem}', common_name=f'n-{alkyl} iodide', smiles=smiles,
             carbon_count=c, functional_group='alkyl iodide', functional_group_structure='-I')
        for stem, alkyl, smiles, c in [
            ('propane', 'propyl', 'CCCI', 3), ('butane', 'butyl', 'CCCCI', 4),
            ('pentane', 'amyl', 'CCCCCI', 5), ('hexane', 'hexyl', 'CCCCCCI', 6),
        ]
    ],

    # --- O/S/N parallels ---
    dict(iupac_name='Methoxyethane', common_name='ethyl methyl ether', smiles='CCOC',
         carbon_count=3, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxypropane', common_name='methyl propyl ether', smiles='CCCOC',
         carbon_count=4, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxybutane', common_name='methyl butyl ether', smiles='CCCCOC',
         carbon_count=5, functional_group='ether', functional_group_structure='-O-'),
    dict(iupac_name='1-Methoxypentane', common_name='methyl pentyl ether', smiles='CCCCCOC',
         carbon_count=6, functional_group='ether', functional_group_structure='-O-'),

    dict(iupac_name='(Methylsulfanyl)ethane', common_name='ethyl methyl sulfide', smiles='CCSC',
         carbon_count=3, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)propane', common_name='methyl propyl sulfide', smiles='CCCSC',
         carbon_count=4, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)butane', common_name='methyl butyl sulfide', smiles='CCCCSC',
         carbon_count=5, functional_group='thioether', functional_group_structure='-S-'),
    dict(iupac_name='1-(Methylsulfanyl)pentane', common_name='methyl pentyl sulfide', smiles='CCCCCSC',
         carbon_count=6, functional_group='thioether', functional_group_structure='-S-'),

    dict(iupac_name='Methyl ethyl sulfoxide', common_name='methyl ethyl sulfoxide', smiles='CS(=O)CC',
         carbon_count=3, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl propyl sulfoxide', common_name='methyl propyl sulfoxide', smiles='CS(=O)CCC',
         carbon_count=4, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl butyl sulfoxide', common_name='methyl butyl sulfoxide', smiles='CS(=O)CCCC',
         carbon_count=5, functional_group='sulfoxide', functional_group_structure='>S=O'),
    dict(iupac_name='Methyl pentyl sulfoxide', common_name='methyl pentyl sulfoxide', smiles='CS(=O)CCCCC',
         carbon_count=6, functional_group='sulfoxide', functional_group_structure='>S=O'),

    dict(iupac_name='Methyl ethyl sulfone', common_name='methyl ethyl sulfone', smiles='CS(=O)(=O)CC',
         carbon_count=3, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl propyl sulfone', common_name='methyl propyl sulfone', smiles='CS(=O)(=O)CCC',
         carbon_count=4, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl butyl sulfone', common_name='methyl butyl sulfone', smiles='CS(=O)(=O)CCCC',
         carbon_count=5, functional_group='sulfone', functional_group_structure='>SO2'),
    dict(iupac_name='Methyl pentyl sulfone', common_name='methyl pentyl sulfone', smiles='CS(=O)(=O)CCCCC',
         carbon_count=6, functional_group='sulfone', functional_group_structure='>SO2'),

    dict(iupac_name='Propan-1-imine', common_name='propionaldehyde imine', smiles='CCC=N',
         carbon_count=3, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Butan-1-imine', common_name='butyraldehyde imine', smiles='CCCC=N',
         carbon_count=4, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Pentan-1-imine', common_name='valeraldehyde imine', smiles='CCCCC=N',
         carbon_count=5, functional_group='imine', functional_group_structure='-CH=NH'),
    dict(iupac_name='Hexan-1-imine', common_name='caproaldehyde imine', smiles='CCCCCC=N',
         carbon_count=6, functional_group='imine', functional_group_structure='-CH=NH'),

    dict(iupac_name='1-Nitropropane', common_name='1-nitropropane', smiles='CCC[N+](=O)[O-]',
         carbon_count=3, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitrobutane', common_name='1-nitrobutane', smiles='CCCC[N+](=O)[O-]',
         carbon_count=4, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitropentane', common_name='1-nitropentane', smiles='CCCCC[N+](=O)[O-]',
         carbon_count=5, functional_group='nitro', functional_group_structure='-NO2'),
    dict(iupac_name='1-Nitrohexane', common_name='1-nitrohexane', smiles='CCCCCC[N+](=O)[O-]',
         carbon_count=6, functional_group='nitro', functional_group_structure='-NO2'),

    # --- Position isomers: group at C2 instead of C1. Merged into the primary
    # alcohol/amine/thiol classes below -- C1 vs C2 diff-vectors measured at
    # ~0.98-0.99 cosine similarity, so position doesn't meaningfully distinguish them. ---
    dict(iupac_name='Propan-2-ol', common_name='isopropanol', smiles='CC(O)C',
         carbon_count=3, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Butan-2-ol', common_name='sec-butanol', smiles='CC(O)CC',
         carbon_count=4, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Pentan-2-ol', common_name='sec-amyl alcohol', smiles='CC(O)CCC',
         carbon_count=5, functional_group='alcohol', functional_group_structure='-OH'),
    dict(iupac_name='Hexan-2-ol', common_name='sec-hexyl alcohol', smiles='CC(O)CCCC',
         carbon_count=6, functional_group='alcohol', functional_group_structure='-OH'),

    dict(iupac_name='Propan-2-amine', common_name='isopropylamine', smiles='CC(N)C',
         carbon_count=3, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Butan-2-amine', common_name='sec-butylamine', smiles='CC(N)CC',
         carbon_count=4, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Pentan-2-amine', common_name='sec-amylamine', smiles='CC(N)CCC',
         carbon_count=5, functional_group='amine', functional_group_structure='-NH2'),
    dict(iupac_name='Hexan-2-amine', common_name='sec-hexylamine', smiles='CC(N)CCCC',
         carbon_count=6, functional_group='amine', functional_group_structure='-NH2'),

    dict(iupac_name='Propane-2-thiol', common_name='isopropyl mercaptan', smiles='CC(S)C',
         carbon_count=3, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Butane-2-thiol', common_name='sec-butyl mercaptan', smiles='CC(S)CC',
         carbon_count=4, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Pentane-2-thiol', common_name='sec-amyl mercaptan', smiles='CC(S)CCC',
         carbon_count=5, functional_group='thiol', functional_group_structure='-SH'),
    dict(iupac_name='Hexane-2-thiol', common_name='sec-hexyl mercaptan', smiles='CC(S)CCCC',
         carbon_count=6, functional_group='thiol', functional_group_structure='-SH'),
]


# ============================
# RDKit: computed properties
# ============================

def compute_rdkit_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return {
        'mw': round(Descriptors.MolWt(mol), 3),
        'tpsa': round(Descriptors.TPSA(mol), 2),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
    }


def compute_avg_carbon_oxidation_state(smiles):
    """Standard organic oxidation-state rule: a bond to H contributes -1 (per
    bond order) to the carbon; a bond to a more electronegative atom
    contributes +1 (per bond order); a bond to another C contributes 0.
    Validated against this dataset's existing values, e.g. propane -> -2.667,
    propanenitrile's nitrile carbon (C#N, triple bond) -> +3.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    if not carbons:
        raise ValueError(f"No carbon atoms in {smiles}")
    states = []
    for c in carbons:
        state = 0
        for bond in c.GetBonds():
            other = bond.GetOtherAtom(c)
            order = int(round(bond.GetBondTypeAsDouble()))
            if other.GetSymbol() == 'H':
                state -= order
            elif other.GetSymbol() == 'C':
                state += 0
            elif other.GetSymbol() in ELECTRONEGATIVE:
                state += order
            else:
                raise ValueError(f"Unhandled neighbor element '{other.GetSymbol()}' bonded to carbon in {smiles}")
        states.append(state)
    return sum(states) / len(states)


# ============================
# PubChem: experimental properties (best-effort)
# ============================

def _pubchem_get(url):
    resp = requests.get(url, timeout=20)
    time.sleep(REQUEST_DELAY_S)
    if resp.status_code != 200:
        return None
    return resp.json()


def resolve_cid(name):
    data = _pubchem_get(PUBCHEM_CID_URL.format(name=requests.utils.quote(name)))
    if not data:
        return None
    try:
        return data['IdentifierList']['CID'][0]
    except (KeyError, IndexError):
        return None


def _iter_info_entries(section_json, heading):
    def walk(node):
        if isinstance(node, dict):
            if node.get('TOCHeading') == heading:
                for info in node.get('Information', []):
                    yield info
            for child in node.get('Section', []):
                yield from walk(child)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
    yield from walk(section_json.get('Record', {}))


def fetch_boiling_point_c(cid):
    data = _pubchem_get(PUBCHEM_VIEW_URL.format(cid=cid, heading='Boiling+Point'))
    if not data:
        return None
    for info in _iter_info_entries(data, 'Boiling Point'):
        value = info.get('Value', {})
        numbers = value.get('Number')
        unit = value.get('Unit', '')
        if numbers and 'C' in unit:
            return float(numbers[0])
    # fall back: parse first free-text string, handle ranges ("84-85 C") and
    # degF -> degC conversion. The minus sign is only treated as a sign when
    # it starts the string/number (not stuck between two digits, which means
    # a "low-high" range, not a negative value).
    for info in _iter_info_entries(data, 'Boiling Point'):
        for s in info.get('Value', {}).get('StringWithMarkup', []):
            text = s.get('String', '')
            for unit, to_celsius in (('F', lambda v: (v - 32) * 5 / 9), ('C', lambda v: v)):
                m = re.search(rf'(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)\s*\xb0\s*{unit}\b', text)
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    return round(to_celsius((lo + hi) / 2), 1)
                m = re.search(rf'(?:^|\s)(-?\d+\.?\d*)\s*\xb0\s*{unit}\b', text)
                if m:
                    return round(to_celsius(float(m.group(1))), 1)
    return None


def fetch_water_solubility(cid):
    data = _pubchem_get(PUBCHEM_VIEW_URL.format(cid=cid, heading='Solubility'))
    if not data:
        return None
    for info in _iter_info_entries(data, 'Solubility'):
        for s in info.get('Value', {}).get('StringWithMarkup', []):
            text = s.get('String', '').lower()
            for keyword, bucket in SOLUBILITY_KEYWORDS:
                if keyword in text:
                    return bucket
    # fall back: bucket a numeric mg/L value if present
    for info in _iter_info_entries(data, 'Solubility'):
        value = info.get('Value', {})
        numbers = value.get('Number')
        unit = value.get('Unit', '')
        if numbers and 'mg/L' in unit:
            mg_per_l = float(numbers[0])
            if mg_per_l >= 1e5:
                return 'high'
            if mg_per_l >= 1e4:
                return 'moderate'
            if mg_per_l >= 1e3:
                return 'low'
            return 'very low'
    return None


# ============================
# Orchestration
# ============================

def build_property_rows(molecules):
    rows = []
    missing = []
    for mol in molecules:
        name, smiles = mol['iupac_name'], mol['smiles']
        print(f"Processing {name} ({smiles})...")

        rdkit_props = compute_rdkit_properties(smiles)
        avg_ox = compute_avg_carbon_oxidation_state(smiles)

        n_carbons = sum(1 for a in Chem.MolFromSmiles(smiles).GetAtoms() if a.GetSymbol() == 'C')
        assert n_carbons == mol['carbon_count'], (
            f"{name}: SMILES has {n_carbons} carbons, expected carbon_count={mol['carbon_count']}"
        )

        cid = resolve_cid(name) or resolve_cid(mol['common_name'])
        boiling_point_c = fetch_boiling_point_c(cid) if cid else None
        water_solubility = fetch_water_solubility(cid) if cid else None
        if cid is None:
            missing.append((name, 'no PubChem CID found'))
        else:
            if boiling_point_c is None:
                missing.append((name, 'no boiling point in PubChem experimental data'))
            if water_solubility is None:
                missing.append((name, 'no water solubility in PubChem experimental data'))

        group = mol['functional_group']
        if group not in MANUAL_PKA:
            raise KeyError(f"No MANUAL_PKA entry for functional group '{group}'")
        pka, pkah = MANUAL_PKA[group]

        rows.append({
            'iupac_name': mol['iupac_name'],
            'common_name': mol['common_name'],
            # Kept as-is for backward compatibility with results already
            # produced from this column. `smiles` and `condensed_formula` are
            # filled in for every row by add_structure_columns; prefer those.
            'formula': smiles,
            'carbon_count': mol['carbon_count'],
            'functional_group': group,
            'functional_group_structure': mol['functional_group_structure'],
            'mw': rdkit_props['mw'],
            'pka': pka,
            'pkah': pkah,
            'tpsa': rdkit_props['tpsa'],
            'avg_carbon_oxidation_state': avg_ox,
            'hbd': rdkit_props['hbd'],
            'hba': rdkit_props['hba'],
            'boiling_point_c': boiling_point_c,
            'water_solubility': water_solubility,
        })
    return rows, missing


def structure_key(iupac_name, functional_group):
    """Which structure template a row uses.

    Only alcohol/amine/thiol are ambiguous: each folds in a 2-substituted
    position isomer alongside the 1-substituted series, and the two need
    different templates. Matching on the locant in the IUPAC name is what
    distinguishes them (`Propan-2-ol` vs `Propan-1-ol`).
    """
    if functional_group in ('alcohol', 'amine', 'thiol'):
        return f"{functional_group}_{'2' if '-2-' in iupac_name else '1'}"
    return functional_group


def parse_condensed_atoms(condensed):
    """Atom counts for one condensed structural formula, hydrogens included."""
    counts, i = {}, 0
    while i < len(condensed):
        if condensed[i] in '()':
            i += 1
            continue
        for fragment, atoms in _CONDENSED_FRAGMENTS:
            if condensed.startswith(fragment, i):
                for atom, k in atoms.items():
                    counts[atom] = counts.get(atom, 0) + k
                i += len(fragment)
                break
        else:
            raise ValueError(
                f"cannot tokenise condensed formula {condensed!r} at {condensed[i:]!r}")
    return counts


def add_structure_columns(df):
    """Populate `smiles` for every row from the templates.

    Applied to the whole combined frame rather than only to NEW_MOLECULES, so
    both notations come from one code path for all 92 molecules. That is the
    point of the fix: the previous split, where generated rows got SMILES and
    original rows kept condensed formulas, is what confounded notation with
    class.
    """
    df = df.copy()
    keys = [structure_key(r.iupac_name, r.functional_group) for r in df.itertuples()]
    missing = {k for k in keys if k not in SMILES_BY_STRUCTURE}
    if missing:
        raise KeyError(f"No structure template for: {sorted(missing)}")
    counts = df['carbon_count'].astype(int).tolist()
    df['smiles'] = [SMILES_BY_STRUCTURE[k](n) for k, n in zip(keys, counts)]
    return df


def verify_structures(df):
    """Both notations must describe the same molecule, for every row.

    The SMILES has to parse, and the condensed form derived from the templates
    has to contain exactly the atoms RDKit finds in it -- hydrogens included,
    which is the part that catches an off-by-one in a CH2 run. Carbon count is
    checked against the existing column too, so a template error cannot survive
    silently. The condensed string is a check only; it is not written out.
    """
    problems = []
    for row in df.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            problems.append(f"{row.iupac_name}: RDKit cannot parse SMILES {row.smiles!r}")
            continue
        rdkit_atoms = {}
        for atom in Chem.AddHs(mol).GetAtoms():
            rdkit_atoms[atom.GetSymbol()] = rdkit_atoms.get(atom.GetSymbol(), 0) + 1
        condensed = CONDENSED_BY_STRUCTURE[
            structure_key(row.iupac_name, row.functional_group)](int(row.carbon_count))
        condensed_atoms = parse_condensed_atoms(condensed)
        if condensed_atoms != rdkit_atoms:
            problems.append(
                f"{row.iupac_name}: condensed form {condensed!r} has {condensed_atoms} "
                f"but SMILES {row.smiles!r} has {rdkit_atoms}")
        n_carbons = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        if n_carbons != int(row.carbon_count):
            problems.append(
                f"{row.iupac_name}: SMILES has {n_carbons} carbons, "
                f"carbon_count says {row.carbon_count}")
    if df['smiles'].isna().any():
        problems.append("some rows have a null smiles")
    if problems:
        raise AssertionError("Structure verification failed:\n  " + "\n  ".join(problems))
    print(f"Verified structures: {len(df)} rows, SMILES present for every molecule and "
          f"atom-for-atom consistent with its condensed form; notation does not vary by class.")


def load_original():
    """Reads the base dataset and drops any rows this script regenerates, so
    re-running the pipeline (ORIGINAL_CSV == OUTPUT_CSV) recomputes those rows
    fresh instead of duplicating them. Matched by iupac_name rather than
    functional_group: some NEW_MOLECULES entries (e.g. the position isomers)
    share a functional_group label with original, non-regenerated rows
    (propan-1-ol and propan-2-ol are both 'alcohol'), so matching on the label
    would incorrectly drop those original rows too.
    """
    df = pd.read_csv(ORIGINAL_CSV)
    df['functional_group'] = df['functional_group'].replace('alkyl halide', 'alkyl chloride')
    df.loc[df['functional_group'] == 'alkyl chloride', 'functional_group_structure'] = '-Cl'
    new_names = {mol['iupac_name'] for mol in NEW_MOLECULES}
    return df[~df['iupac_name'].isin(new_names)]


def verify(df):
    # Merged classes (alcohol/amine/thiol, which fold in the C2 position
    # isomers) legitimately have >1 molecule at the same chain length -- the
    # real invariant is just that no chain length (3-6) is missing entirely,
    # not that every class has exactly 4 rows.
    problems = []
    for group, sub in df.groupby('functional_group'):
        if group == 'none (alkane)':
            continue
        missing = {3, 4, 5, 6} - set(sub['carbon_count'])
        if missing:
            problems.append(f"'{group}' is missing chain length(s) {sorted(missing)}")
    if problems:
        raise AssertionError("Dataset verification failed:\n  " + "\n  ".join(problems))
    print(f"Verified: {len(df)} rows, {df['functional_group'].nunique()} functional-group classes, "
          f"each non-alkane class covers chain lengths 3-6.")


def main():
    # The structure columns depend only on (iupac_name, functional_group,
    # carbon_count), so they can be (re)generated without touching PubChem.
    # That matters: a full run re-fetches boiling point and water solubility for
    # every molecule, and those are best-effort scrapes whose values can change
    # or go missing. This mode adds the notation columns to the existing file
    # and leaves every other column exactly as it is.
    if '--structure-columns-only' in sys.argv:
        df = pd.read_csv(ORIGINAL_CSV)
        df = add_structure_columns(df)
        verify_structures(df)
        verify(df)
        df[COLUMNS].to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote {len(df)} rows to {OUTPUT_CSV} "
              f"(structure columns only; no network calls, no other column touched)")
        return

    original = load_original()

    new_rows, missing = build_property_rows(NEW_MOLECULES)
    new_df = pd.DataFrame(new_rows)

    combined = pd.concat([original, new_df], ignore_index=True)
    combined = add_structure_columns(combined)[COLUMNS]
    verify(combined)
    verify_structures(combined)

    if missing:
        print(f"\n=== NEEDS MANUAL FOLLOW-UP ({len(missing)} gaps) ===")
        for name, reason in missing:
            print(f"  {name}: {reason}")

    na_boiling = combined['boiling_point_c'].isna().sum()
    na_solubility = combined['water_solubility'].isna().sum()
    print(f"\nboiling_point_c missing for {na_boiling}/{len(combined)} rows "
          f"(PubChem had no experimental data for those compounds)")
    print(f"water_solubility missing for {na_solubility}/{len(combined)} rows")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(combined)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
