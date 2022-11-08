# 2D_hidden_elasticity_PINN
Data and codes from our 2022 paper: 

[Elasticity Imaging Using Physics-Informed Neural Networks: Spatial Discovery of Elastic Modulus and Poisson's Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110)
Preprint link: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110)

## Quick Setup

install [SCiANN](https://www.sciann.com/) on a PC or virtual machine using:

```
pip install sciann
```
## Running the PINN code
Clone this repository and navigate to the Embedded_Inclusion_PlaneStress directory. Running the parameter discovery code could be done using the following function
for a batch size of 1024, 200,000 training epochs:

```
python PE_PINN.py -bs 1024 -e 200000

```

The code takes in 1D txt files containing x and y collocation points and strain terms as input and returns distributions of Young's modulus, Poisson's ratio, and stress at those coordinates.


## Plotting
One can use the included helper function to plot unstructured data over a structured grid given desired number of points on the x axis
```
python Plot_PE_PINN_Results.py -nxp 50
```

Sample input and output files for a 200,000 run are included as sample data.
Have a question about implementing the code? contact us at [klaksari@arizona.edu](mailto:klaksari@arizona.edu), akamali@arizona.edu(mailto:akamali@arizona.edu)
