import numpy as np

for idx in range(1,4000): 
    ir = []
    raman = []
    try:
        with open(f'/home/yanggk/Data/H2L_Data/tsps/Au/IR/clear/{idx}.txt','r') as f:
            aaa = f.readlines()
            for i in aaa:
                i = i.replace('\n','').split(',')
                while '' in i:
                    i.remove('')
                ir.append(i[1])
            # ir = np.array(ir)

            # with open(f'/home/yanggk/Data/HW/Raman/ori/curve/{idx}.txt','r') as ra:
            #     bbb = ra.readlines()
            #     for i in bbb:
            #         i = i.replace('\n','').split(',')
            #         while '' in i:
            #             i.remove('')
            #         raman.append(i[1])
            # raman = np.array(raman)
        with open('/home/yanggk/Data/H2L_Data/tsps/Au/IR/ts_formed_1cm-1_clr.csv','a') as w:
            print(idx,'ir',len(ir),'raman',len(raman))
            w.write(str(idx) + ',' + ','.join(ir) + '\n')
    except:pass