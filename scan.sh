#!/bin/bash
lr=("0.001"\
   "0.01"\
   )
bs=("16"\
   "32"\
   "64"\
   "128"\
   "256"\
   "512"\
)

for a in "${lr[@]}"
do
for b in "${bs[@]}"
do 
echo 'Conducting:' ${a}_${b}
python unet_attn.py --a $a --b $b  > logs/${a}_${b}.log
cp -r spec ~/Data/HW/neo_Raman_spec_${a}_${b}
cp -r figs ~/Data/HW/neo_Raman_figs_${a}_${b}
cp -r spec_data ~/Data/HW/neo_Raman_sepc_data_${a}_${b}
echo 'Done'
done
done
