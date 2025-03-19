'''
This script is written by XIANG XING (xxiangad@connect.ust.hk)
to calculate thermal conductivity based on phonon BTE theory.
The script aims to automatically finish calculations, once 
primitive structure is given. 
'''


import os
import shutil
from glob import glob

import numpy as np
import h5py

from ase.io import read, write
#from calorine.calculators import CPUNEP
#from calorine.tools import relax_structure
from matplotlib import pyplot as plt

from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from shutil import copyfile
from ase.dft.kpoints import *


# set work directory
root_dir = os.getcwd()  # here we set the root directory
work_dir = os.path.join(root_dir, '1_prim_relax')
shutil.rmtree(work_dir, ignore_errors=True)  # Deletes current working dir
os.mkdir(work_dir)
os.chdir(work_dir)

print(f'Root directory: {root_dir}')
print(f'Working directory: {work_dir}')

# relax structures
#potential_filename = os.path.join(root_dir, 'nep-graphite-CX.txt')
prim = read(os.path.join(root_dir, 'POSCAR_initial'))

#prim.calc = CPUNEP(potential_filename)
#relax_structure(prim, fmax=1e-5)

env_pot ='/home/user/Desktop/Software/vasp/vasp_6.4/pesdopotential'
os.environ['VASP_PP_PATH']=env_pot

#mydir_relax = os.path.join(dir,'relax')
calc = Vasp(command= 'mpirun -np 120  /home/user/Desktop/Software/vasp/vasp_6.4/vasp.6.4.3/bin/vasp_std > log' ,
            prec='Accurate',
            isif=3,
            isym=2,
            ibrion=2,
            nsw=500,
            ediff=1e-5,
            ediffg=-0.01,
            encut=600,
            lwave=False,
            lcharg=False,
            ismear=0,
            sigma = 0.05,
            xc='PBE',
            lreal=False,
            kpts=(9,9,9),
            ncore=20,
            directory=work_dir)
            
prim.calc = calc
prim.get_potential_energy()

# construction of force constant
work_dir = os.path.join(root_dir, '2_force_constant')
shutil.rmtree(work_dir, ignore_errors=True)  # Deletes current working dir
os.mkdir(work_dir)
os.chdir(work_dir)

print(f'Working directory: {work_dir}')
# supercell size
write('POSCAR', prim,direct=True, sort=True)

dim = (2, 2, 2)

cmd = f'phono3py -d --dim="{dim[0]} {dim[1]} {dim[2]}" --dim-fc2="{dim[0]} {dim[1]} {dim[2]}"'

print(f'Running command: {cmd}')
os.system(cmd)

fnames = sorted(glob('POSCAR_FC2-*'))
print(fnames)
forces_data = []
for it, fname in enumerate(fnames):
    structure = read(fname)

    structure.calc = Vasp(command= 'mpirun -np 120  /home/user/Desktop/Software/vasp/vasp_6.4/vasp.6.4.3/bin/vasp_std > log' ,
            prec='Accurate',
            isif=2,
            isym=2,
            ibrion=-1,
            nsw=0,
            encut=600,
            lwave=False,
            lcharg=False,
            ismear=0,
            sigma = 0.05,
            xc='PBE',
            lreal=False,
            kpts=(1,1,1),
            ncore=20,
            directory=work_dir)
    
    forces = structure.get_forces()
    forces_data.append(forces)
    print(f'FC2: Calculating supercell {it} / {len(fnames)}, f_max {np.max(np.abs(forces)):8.5f}')
forces_data = np.array(forces_data).reshape(-1, 3)
np.savetxt('FORCES_FC2', forces_data)

fnames = sorted(glob('POSCAR-*'))
print(fnames)
forces_data = []
for it, fname in enumerate(fnames):
    #print(it)
    structure = read(fname)

    structure.calc = Vasp(command= 'mpirun -np 120  /home/user/Desktop/Software/vasp/vasp_6.4/vasp.6.4.3/bin/vasp_std > log' ,
            prec='Accurate',
            isif=2,
            isym=2,
            ibrion=-1,
            nsw=0,
            encut=600,
            lwave=False,
            lcharg=False,
            ismear=0,
            sigma = 0.05,
            xc='PBE',
            lreal=False,
            kpts=(1,1,1),
            ncore=20,
            directory=work_dir)
    
    forces = structure.get_forces()
    forces_data.append(forces)
    print(f'FC3: Calculating supercell {it} / {len(fnames)}, f_max= {np.max(np.abs(forces)):8.5f}')
forces_data = np.array(forces_data).reshape(-1, 3)
np.savetxt('FORCES_FC3', forces_data)

cmd = f'phono3py --cfc --hdf5-compression gzip -c phono3py_disp.yaml'
print(f'Running command: {cmd}')
os.system(cmd)

# BTE calculations
work_dir = os.path.join(root_dir, '3_BTE_calculations')
shutil.rmtree(work_dir, ignore_errors=True)  # Deletes current working dir
os.mkdir(work_dir)
shutil.copy('fc2.hdf5', work_dir) 
shutil.copy('fc3.hdf5', work_dir) 
shutil.copy('phono3py_disp.yaml', work_dir) 
os.chdir(work_dir)

print(f'Working directory: {work_dir}')

mesh = [16, 16, 16]  # q-point mesh
T_min, T_max = 10, 1000  # temperature range
T_step = 10  # spacing of temperature points

cmd = f'phono3py -q --fc2 --fc3 --bterta --dim="{dim[0]} {dim[1]} {dim[2]}" --dim-fc2="{dim[0]} {dim[1]} {dim[2]}"'\
      f' --tmin={T_min} --tmax={T_max} --tstep={T_step} --mesh "{mesh[0]} {mesh[1]} {mesh[2]}"'
print(f'Running command: {cmd}')

os.system(cmd)
