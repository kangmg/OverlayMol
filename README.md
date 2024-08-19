# OverlayMol
Overlay and align molecular structures for comparison and analysis.

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
