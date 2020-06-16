Important:
Since my version of matplotlib does not support saving as jpg directly
I save it as png and use PIL to convert it to jpg

In ex3 and ex4, there are some functions to calculate Mutual Information based scores 
e.g. adjusted_mutual_info_score(), mutual_info_score() and  normalized_mutual_info_score()
I use mutual_info_score() which is possible to exceed 1.0

In ex4, when calculating silhouette_avg, I have not enough memory to calculate. 
Therefore, I specific sample size is 1/3 of original one.

Enviroment:
Python 3.5.3
Package         Version
--------------- ------------
imageio         2.8.0
kiwisolver      1.1.0
matplotlib      3.0.3
numpy           1.18.2
Pillow          7.0.0
pip             20.0.2
scikit-learn    0.22.2.post1
scipy           1.4.1