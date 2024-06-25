# TriCorreX
Any practioner and/or enthasuasist of molecular dynamics has likely come across the radial distrubtion function (RDF). The RDF is a simple yet robust way to characterize a system: compares the configuration of a particles local enviroment to their configuration in a perfect gas. In a perfect gas the particles behave at random and show no perference in regards to distance seperation. But for interacting particles, if the potential in use provides attractive distances, those distances will find on average higher particle counts, like below for the classic Lennard-Jones (LJ) fluid. 

![Pair Analysis](images/rdf.png "Pair Analysis Diagram")
This reveals the perferred 2-body seperation distrubtion at $\rho^{*}=0.45$ and $T^{ *}=1.0$. With a maximum at $\approx 1.12 \ \sigma$ 

The situation complicates itself when it comes to 3-body interactions. While 2-body interactions can be characterized by *one* pair seperation distance, 3-body interactions require accounting for 3 independent pair seperation distances. 


