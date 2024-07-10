# TriCorreX
Any practioner and/or enthasuasist of molecular dynamics has likely come across the radial distrubtion function (RDF). The RDF is a simple yet robust way to characterize a system: compares the configuration of a particles local enviroment to their configuration in a perfect gas. In a perfect gas the particles behave at random and show no perference in regards to distance seperation. But for interacting particles, if the potential in use provides attractive distances, those distances will find on average higher particle counts, like below for the classic Lennard-Jones (LJ) fluid. 

![Pair Analysis](images/rdf.png "Pair Analysis Diagram")
This reveals the perferred 2-body seperation distrubtion at $\rho^{*}=0.45$ and $T^{ *}=1.0$. With a maximum at $\approx 1.12 \ \sigma$. In practice, the RDF is approximated by dividing the radial distance of interest into seperate bins, and couting how many pair seperation distances fall within each bin (i.e. how many pairs of particles were seperated by a distance $d \in \lbrace r-\Delta r / 2, \ r+\Delta r / 2 \rbrace$). The choice of $\Delta r$ leads to finer or coarser binning, which can *potentially* lead to problems down the line. However, thoughtful smoothing can mitigate if not eliminate these issues. In general, measuring the RDF is safe in that their are few if any suprises in it's approximation.  

The situation complicates itself when it comes to 3-body interactions. While 2-body interactions can be characterized by *one* pair seperation distance, 3-body interactions require accounting for 3 independent pair seperation distances. For a pair ijk with seperation distances $r_{\text{ij}}$, $r_{\text{ik}}$, $r_{\text{jk}}$ we now need a bin accounting for a triplet centered at some 3 distances such that the bin width captures all 3. If we needed N bins for the RDF approximation, we now need $\text{N}^3$ bins to adequetly measure the 3-body correlations. This quickly reminds us of the curse of dimensionality, where far more data is needed as the dimensionality of the underlying distribution is increased. 

Sufficient sampling in either case is essential for a trust-worthy approximation to the true correlation function. In the case of the RDF, if we have N data points for the M bins, our (best case) average error scales as: 
$$\propto \sqrt{\frac{\text{N}}{\text{M}}}^{-1}$$
An equivalent 3-body histogram would require $M^3$ bins, and as such it's error per bin scales as:
$$\propto \sqrt{\frac{\text{N}}{\text{M}^3}}^{-1}$$
And to achieve the same statisical error as the RDF, our 3-body histogram would require $M^2$ more data-points.  

Within a volume V at number density $\rho$ we have $\text{p}=\rho \text{V}$ particles, and thus $\text{p}(\text{p}-1)/2$ unique particle pairs, and 
$\text{p}(\text{p}-1)(\text{p}-2)/6$ unique triplets. If we approximated the RDF with one configuration using M bins, then our error would scale as:
$$\propto \sqrt{\frac{\text{p}(\text{p}-1)}{2\text{M}}}^{-1}$$
To get the same error in the 3-body histogram would require: 

$$\text{M}^2\text{p}(\text{p}-1)/2$$ 

points. Or $3\text{M}^2/(\text{p}-2)$ as many configurations. For typical MD systems to mitigate finite size effects we might consider $10^4$ particles, if we desired M=300 then we would require ~27 configurations to achieve the same error as one configuration for the RDF. Clearly, algorithms that efficiently locate and count triplets are essential for computationally feasible approximations of the 3-body correlation function.  


