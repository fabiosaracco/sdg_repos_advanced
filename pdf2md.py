import os, sys, pickle, json
import pathlib
import numpy as np
from datetime import datetime as dt

from tqdm.auto import tqdm, trange
from multiprocessing import Pool

import pymupdf4llm
import fitz



from file_handler import *

PDF_FOLDER='./all_pdfs/'
MD_FOLDER='./all_mds/'
pdfs=[file for file in os.listdir(PDF_FOLDER) if file.endswith('.pdf')]
l_pdfs=len(pdfs)

def main():
    
    
    pdfs.sort()
    
    with Pool(6) as p:
        errors=p.map(pdf2md_singlestep, pdfs)
        
    errors=[e[0] for e in errors if len(e)>0] 
    with open(MD_FOLDER+'errors.jsonl', 'w') as fp:
        json.dump(errors, fp)

def pdf2md_singlestep(pdf):
    os.environ["PYMUPDF_MESSAGE"] = ""  # silenzia i messaggi PyMuPDF/Tesseract a monte
    errors=[]
    
    mds=[file for file in os.listdir(MD_FOLDER) if file.endswith('.md')]
    l_mds=len(mds)
    
    _message=f'[{dt.now():%y-%m-%d %H:%M:%S}] {l_mds*100//l_pdfs:3d}% {pdf}'
    
    print(_message)
    sys.stdout.flush()
    try:
        # check if the original file has some issue
        fitz.TOOLS.reset_mupdf_warnings()
        doc = fitz.open(PDF_FOLDER+pdf)

        if doc.is_repaired:
            errors.append({"file": pdf, "type": "repaired",
                        "detail": fitz.TOOLS.mupdf_warnings()})

        # check if the file was already processed
        file_name=file2vecfile(pdf, extension='.pdf').replace('.txt', '.md')
        if os.path.exists(MD_FOLDER+file_name):
            return errors

        md_text = pymupdf4llm.to_markdown(PDF_FOLDER+pdf, show_progress=False)

        # check if the file is too small (thus, it has ome problem)
        n_paragraphs = md_text.count("\n\n")
        if len(md_text) < 500 or n_paragraphs < 3:
            errors.append({"file": pdf, "type": "suspiciously_short_or_flat",
                        "chars": len(md_text), "paragraphs": n_paragraphs})
        else:
            pathlib.Path(MD_FOLDER+file_name).write_bytes(md_text.encode())

    except (fitz.EmptyFileError, fitz.FileDataError) as e:
        errors.append({"file": pdf, "type": "hard_failure", "detail": str(e)})
    except Exception as e:
        errors.append({"file": pdf, "type": "unexpected", "detail": str(e)})
    return errors
        
        
if __name__ == "__main__":
    main()
    os.system("afplay ta-da.mp3")
        
    