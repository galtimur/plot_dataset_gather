import nbformat as nbf

def build_new_nb(blocks: list, nb_path):

    '''
    save codeblocks into notebook
    '''

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(block) for block in blocks]

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)