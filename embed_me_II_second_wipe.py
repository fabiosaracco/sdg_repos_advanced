import os, sys, pickle
import numpy as np

from datetime import datetime as dt
import traceback

from sentence_transformers import util

from tqdm.auto import tqdm, trange

from file2text import file2text

from file_handler import *

from jina4lote import jina4lote

RTF_FOLDER='./rtf/'
TARGET_FOLDER='./NewProcessedData/'

def main():
    # first, get the files
    files=os.listdir(RTF_FOLDER)
    files=[file for file in files if file.endswith('.rtf')]
    files.sort()
    i_f=0
    while i_f<len(files):
        file=files[i_f]
        
        # get name for the output
        vecfile=file2vecfile(file)
        if not os.path.isfile(TARGET_FOLDER+vecfile):
            print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:30}')
            # check if also the next file has the same output:
            file1=files[i_f+1]
            vecfile1=file2vecfile(file1)
            if vecfile!=vecfile1:
                i_f+=1
                # if there is no "second" file, I can directly get the embeddings
                try:
                    # extract the text from the .rtf file
                    long_text=file2text(RTF_FOLDER+file)
                    # embed the text using Jina
                    vec=jina4lote(long_text)
                except:
                    _err_msg=traceback.format_exc()
                    print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:30}\tError!')
                    print(_err_msg)
                    continue
            else:
                i_f+=2
                try:
                    # extract the text from the .rtf file
                    long_text=file2text(RTF_FOLDER+file1)
                    # embed the text using Jina
                    vec=jina4lote(long_text)
                except:
                    _err_msg=traceback.format_exc()
                    print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:30}\tError!')
                    print(_err_msg)
                    continue
            np.savetxt(TARGET_FOLDER+vecfile, vec)
        else:
            i_f+=1

            
            

if __name__ == "__main__":
    main()