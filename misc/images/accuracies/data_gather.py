import json


def _compile_filepath_list():
        import glob, os
        filepaths = []
        for root, dirs, files in os.walk("./"):
            for file in files:
                if file.endswith(".json"):
                    filepath = os.path.join(root,file)
                    filepaths += [filepath]
        return filepaths

filepaths = _compile_filepath_list()

for fp in filepaths:
    with open(fp) as infile:
        contents = json.load(infile)
    print "File is {0}           maximum validation accuracy was           {1}.".format(fp, max(contents['val_acc'].values()))
