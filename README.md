# Urban Economic Fitness and Complexity from Patent Data
Data and codes of the paper "Urban Economic Fitness and Complexity from Patent Data"
https://www.nature.com/articles/s41598-023-30649-1

Bipartite Configuration Model: https://github.com/mat701/BiCM

The matrices in Data are ordered by row and column according to the lists of Metropolitan Areas and Technology codes provided.

In the codes.py file, you can find:

  1. rca(biadjm): code to compute RCA of a given bi-adjacency (biadjm) numpy array;
  2. Fitn_Comp(biadjm): code to compute the Fitness and the Complexity, starting from a bi-adjacency (biadjm) numpy array. N.B. it's important for the calculation to delete all the rows and columns with all 0s;
  3. coherence(biadjm,B_network): code to compute the coherence diversification of the MAs, starting from a given bi-adjacency (biadjm) numpy array and the technology network (B_network).


How to cite:
Straccamore, Matteo, et al. "Urban economic fitness and complexity from patent data." Scientific Reports 13.1 (2023): 3655.
