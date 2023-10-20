
for idx in range(1,3501):
    with open(f'{idx}.spec','r') as f:
        print(f'Handling {idx}')
        aaa = f.readlines()
        Freq , IR, Raman = [],[],[]
        for i in aaa:
            a =  i.replace('\n','').split(' ')
            while '' in a:
                a.remove('')
            if len(a) <= 4:
                pass
            else:
                if a[0] == 'Frequencies':
                    Freq.append(a[-3])
                    Freq.append(a[-2])
                    Freq.append(a[-1])
                elif a[0] == 'IR':
                    IR.append(a[-3])
                    IR.append(a[-2])
                    IR.append(a[-1])
                elif a[0] == 'Raman':
                    Raman.append(a[-3])
                    Raman.append(a[-2])
                    Raman.append(a[-1])

        # print(len(Freq))
        # print(len(IR))
        # print(len(Raman))

        with open(f'{idx}.tsv','a') as w:
            print(f'Writing {idx}')
            w.write('idx\tFreq\tIR\tRaman\n')
            for n in range(len(Freq)):
                w.write(str(n)+'\t'+Freq[n]+'\t'+IR[n]+'\t'+Raman[n]+'\n')