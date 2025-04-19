def file2ysn(file, extension):
    '''
    From the file name return, respectively, the year, the sector and the name 
    '''
    _split=file.replace(extension, '').split('_')
    #if ' ' in _split[-1] and len(_split[-1])>4:
    #    return _split[-1].split(' ')[0], _split[0].zfill(2), _split[-2]
    if '(' in _split[-1] or len(_split[-1])<4:
        return _split[-2], _split[0].zfill(2), _split[-3]
    else:
        return _split[-1], _split[0].zfill(2), _split[-2]
    
def say_my_name(name):
    return name.replace(' ', '_')

def vec_file_name(year, sector, name):
    myname=say_my_name(name)
    return year+'_'+sector+'_'+myname+'.txt'

def file2vecfile(file, extension='.rtf'):
    y, s, n=file2ysn(file, extension)
    return vec_file_name(y, s, n)