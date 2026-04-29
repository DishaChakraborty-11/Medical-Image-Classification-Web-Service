# medical_insights.py - Clinical Database
TUMOR_INFO = {
    'glioma': {
        'early_symptoms': [
            'Persistent headaches (worse in morning)',
            'Nausea/vomiting',
            'New-onset seizures',
            'Neurological deficits (weakness, speech)',
            'Personality/memory changes',
            'Vision problems'
        ],
        'diagnosis': [
            'MRI: infiltrative, heterogeneous enhancement',
            'MRS: ↑choline, ↓NAA',
            'Perfusion MRI: variable',
            'Stereotactic biopsy (gold standard)',
            'Advanced: PET/SPECT'
        ],
        'prognosis': 'Grade-dependent (II-IV), aggressive'
    },
    'meningioma': {
        'early_symptoms': [
            'Headaches (mild-moderate)',
            'Seizures',
            'Focal neurological deficits',
            'Vision changes',
            'Hearing loss'
        ],
        'diagnosis': [
            'MRI: well-circumscribed, dural-based',
            '"Dural tail" sign (80%)',
            'Homogeneous enhancement',
            'CT: calcifications common',
            'Usually WHO Grade I (benign)'
        ],
        'prognosis': 'Excellent (90% cure with surgery)'
    },
    'pituitary': {
        'early_symptoms': [
            'Bitemporal hemianopsia (vision loss)',
            'Headaches',
            'Hormonal: Cushing/acromegaly',
            'Menstrual irregularities',
            'Fatigue/weight gain'
        ],
        'diagnosis': [
            'MRI: sellar/suprasellar mass',
            'Endocrine panel (PRL, ACTH, GH)',
            'Visual field testing',
            'Dynamic contrast MRI',
            'Inferior petrosal sinus sampling'
        ],
        'prognosis': 'Good (surgery + meds/radiation)'
    },
    'no_tumor': {
        'early_symptoms': ['None (normal findings)'],
        'diagnosis': ['Normal MRI brain'],
        'prognosis': 'Normal - no intervention needed'
    }
}
