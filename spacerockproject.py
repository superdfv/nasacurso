

# strPath = 'rocks.txt'
# fileObject = open(strPath, 'w')
# fileObject.writelines(["Reading Rocks\n","basalt\n","breccia\n","highland\n","regolith\n",
#                         "highland\n","breccia\n","highland\n","regolith\n","regolith\n",
#                         "basalt\n","highland\n","basalt\n","breccia\n","breccia\n","regolith\n",
#                         "breccia\n","highland\n","highland\n","breccia\n","basalt\n" ])
# fileObject.close()

def count():
    strPath = 'rocks.txt'
    fileObject = open(strPath)
    rock_list = fileObject.readlines()

    basalt = 0
    breccia = 0
    highland = 0
    regolith = 0

   
    for rock in rock_list:
        if rock == 'basalt\n':
            basalt += 1
        elif rock == 'breccia\n':
            breccia += 1
        elif rock == 'highland\n':
            highland += 1
        elif rock == 'regolith\n':
            regolith += 1
    print(f"\nbasalt: {basalt}\t breccia: {breccia}\t highland: {highland}\t regolith: {regolith}")

    fileObject.close()
    
    
count()






   





