from striprtf.striprtf import rtf_to_text

from first_rough_cleaning import first_text_cleaner

def file2roughtext(file):
    with open(file, 'r') as f:
        rtf=f.read()
    return rtf_to_text(rtf)

def file2text(file):
    return first_text_cleaner(file2roughtext(file))