import pandas as pd

for idx in range(1,4000):
    try:
        aaa = pd.read_csv(f'/home/yanggk/Data/H2L_Data/tsps/Au/spec/{idx}.tsv',delimiter='\t')
        # print(a)
        print(len(aaa))
        # print(a['ir'])
        with open(f'/home/yanggk/Data/H2L_Data/tsps/Au/IR_inp/{idx}.inp','w') as wIR:
            wIR.write('72\t1\n')
            for i in range(len(aaa)):
                wIR.write(str(aaa['Freq'][i])+'\t'+ str(aaa['IR'][i])+'\n')
        with open(f'/home/yanggk/Data/H2L_Data/tsps/Au/Raman_inp/{idx}.inp','w') as wRaman:
            wRaman.write('72\t1\n')
            for i in range(len(aaa)):
                wRaman.write(str(aaa['Freq'][i])+'\t'+ str(aaa['Raman'][i])+'\n')
    except: pass

