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
        print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:30}')
        if not os.path.isfile(TARGET_FOLDER+vecfile):
            try:
                # extract the text from the .rtf file
                long_text=file2text(RTF_FOLDER+file)
                # embed the text using Jina
                vec=jina4lote(long_text)
            except:
                _err_msg=traceback.format_exc()
                print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:30}\tError!')
                print(_err_msg)
                i_f+=1
                continue
            # check if also the next file has the same output:
            file1=files[i_f+1]
            vecfile1=file2vecfile(file1)
            if vecfile==vecfile1:
                try:
                    # extract the text from the .rtf file
                    long_text1=file2text(RTF_FOLDER+file1)
                    # embed the text using Jina
                    vec1=jina4lote(long_text1)
                except:
                    _err_msg=traceback.format_exc()
                    print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile1:100} Error!')
                    print(_err_msg)
                    print('Taking the first embedding.')
                    np.savetxt(TARGET_FOLDER+vecfile, vec)
                    i_f+=2
                    continue
                _cs=float(util.cos_sim(vec, vec1))
                print(f'[{dt.now():%y-%m-%d %H:%M:%S}] {vecfile:100}, {vecfile1:100}, cs={_cs:.3f}')
                if 1-_cs>10**-3:
                    print('Such a situation should be handed properly, postponing the issue to a later moment')
                else:
                    print('The difference is limited, taking the former vector.')
                    np.savetxt(TARGET_FOLDER+vecfile, vec)
                i_f+=2
            else:
                np.savetxt(TARGET_FOLDER+vecfile, vec)
                i_f+=1
        else:
            i_f+=1

            
            



if __name__ == "__main__":
    main()