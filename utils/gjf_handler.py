class GJF_handler():
    def __init__(self,file_path) -> None:
        self.file_path = file_path
        self.idx = []
        with open(self.file_path,'r') as f:
            cont = f.readlines()
            for n,line in enumerate(cont):
                if line == '\n':
                    self.idx.append(n)
            self.head = cont[:self.idx[0]]
            self.title = cont[self.idx[0]+1:self.idx[1]]
            self.chg_selfspin = cont[self.idx[1]+1:self.idx[1]+2]
            self.coord = cont[self.idx[1]+2:self.idx[2]]
            self.tails = cont[self.idx[2]+1:]

    def wirte_file(self,name):
        with open(f'{name}','w') as w:
            w.writelines(self.head)
            w.write('\n')
            w.writelines(self.title)
            w.write('\n')
            w.writelines(self.chg_selfspin)
            w.writelines(self.coord)
            w.write('\n')
            w.writelines(self.tails)

    def _z_filter(self):
        item_lis = []
        for item in self.coord:
            if eval(item[-10]+item[-9]) >= 13 :
                neo_item = []
                if item[1]+item[2] == 'Au':
                    for n,char in enumerate(item):
                        if n == 5: 
                            neo_item.append('-')
                        elif n ==6:
                            neo_item.append('1')
                        else:neo_item.append(char)
                    neo_item = ''.join(neo_item)
                    item_lis.append(neo_item)
                else:item_lis.append(item)

        self.coord = item_lis
    
    def del_atom(self):
        self._z_filter()




# a = GJF_handler('Au_BPE_vert000001.gjf')
# # print(a.head)
# # print(a.title)
# # print(a.chg_selfspin)
# # print(a.coord)
# # for item in a.coord:
# #     print(item[-10]+item[-9])
# # for item in a.coord:
# #     if item != '\n':
# #         print(item[1]+item[2])
# a.del_atom()
# a.wirte_file('test.gjf')

if __name__ ==  '__main__':
    for idx in range(1,4002):
        a = GJF_handler(f'gjfs/test ({idx}).gjf')
        a.del_atom()
        a.wirte_file(f'neo_gjfs/{idx}.gjf')