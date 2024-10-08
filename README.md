# OverlayMol
Overlay and align molecular structures for comparison and analysis.

<br>

## Installation
> pip
```bash
pip install -q git+https://github.com/kangmg/OverlayMol.git
```
> git clone
```bash
git clone https://github.com/kangmg/OverlayMol.git
cd OverlayMol
pip install .
```

<br>

## Overlay Diagram

```python
!wget -q https://raw.githubusercontent.com/kangmg/OverlayMol/main/examples/DA.xyz -O DA.xyz

from overlaymol import OverlayMolecules

# config molecules to overlay
DA = OverlayMolecules()
DA.set_molecules('./DA.xyz')

# check default parameters
display(DA.parameters)

# change parameter options
DA.parameters.legend = True
DA.parameters.atom_scaler = 300
DA.parameters.bond_scaler = 8200000
DA.parameters.alpha_bonds = 1
DA.parameters.alpha_atoms = 1


# plot overlay diagram
DA.plot_overlay()
```
![newplot](https://github.com/user-attachments/assets/7915d4cf-b80a-4c69-b874-07b0483561ee)

<br>

## Traj animation

```python
!wget -q https://raw.githubusercontent.com/kangmg/aimDIAS/main/examples/wittig.xyz -O wittig.xyz

from overlaymol import OverlayMolecules

# config molecules to overlay
wittig = OverlayMolecules()
wittig.set_molecules('./wittig.xyz')

# check default parameters
display(wittig.parameters)

# change parameter options
wittig.parameters.colorby = 'molecule'
wittig.parameters.superimpose_option = 'aa'

# plot traj animation
wittig.plot_animation()
```
![newplot (2)](https://github.com/user-attachments/assets/2743b50f-0992-4617-97a7-df87563d48fd)

## Low level API

```python
!wget -q https://github.com/kangmg/OverlayMol/blob/main/examples/sn2.xyz

from overlaymol import open_xyz_files, plotly_overlay, superimpose

# config molecules to overlay
molecules_jsons = open_xyz_files('./sn2.xyz')

# superimpose
superimposed_molecules_json = superimpose(molecules_jsons)

# plot overlay diagram
plotly_overlay(
    xyz_format_jsons=superimposed_molecules_json, 
    colorby='atom',
    bgcolor='navy')
```
![newplot (3)](https://github.com/user-attachments/assets/bde76a8b-e67b-46b9-b699-ddb80e3cfde3)

