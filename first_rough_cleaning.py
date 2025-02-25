def first_text_cleaner(text):
    _split=split_and_brutal_clean(text)
    _split=remove_indices(_split)
    _split=delete_duplicates(_split)
    _split=delete_titles(_split)
    return ' '.join(_split)

def split_and_brutal_clean(text):
    '''
    Remove tabs and useless blank entries
    '''
    no_t_text=text.replace('\t', '')
    return [split.strip() for split in no_t_text.split('\n') if len(split.strip())>0]

def remove_indices(splitted_text):
    '''
    From the first split remove the entries related to indices
    '''
    out=[]
    for split in splitted_text:
        if not split[0].isdigit(): 
            # if either the first character is not a digit,
            # and therefore is not a title or the entry of an index,
            # take it
            out.append(split)
        elif (split[0].isdigit() and ('.' in split or ',' in split)):
            # or if it is part of a sentence
            out.append(split)
    return out

def delete_duplicates(splitted_text):
    '''
    Delete too frequent sentences (they do not carry any information)
    '''
    st, k_st=np.unique(splitted_text, return_counts=True)
    freq_dict=dict(zip(st, k_st))
    return [split for split in splitted_text if freq_dict[split]<=2]

def delete_titles(splitted_text):
    '''
    Titles start with a capital letter and do not end with a punctuation mark.
    Moreover, they do not carry any relevant info about the text.
    '''
    out=[]
    counter=0
    while counter<len(splitted_text):
        text=splitted_text[counter]
        if text[0].islower(): 
            # a smaller character indicates it is the end of a previous sentence
            out.append(text)
        elif any([point in text for point in '.!?']) or splitted_text[counter+1][0].islower():
            # we can have either the case in which a sentence is present, and therefore we have a punctuation,
            # or the case in which the sentence is ended in the following line
            out.append(text)
        counter+=1
    return out