#CECS

CECS(Individual entity induced label concept set for classification: An information fusion viewpoint)

Abstract:

Formal concept analysis has seldom been employed for classification.
This is mainly due to 1) the high time and space complexity of concept lattice construction, and
2) the difficulty of concept lattice based prediction.
Inspired by information fusion, this paper introduces a new algorithm named CECS, which constructs a concept set instead of a lattice to ensure efficiency and enable direct classification.
Regarding concept set construction, we define sub-formal context by grouping objects with the same label within the labeled formal context.
Subsequently, we induce the label concept from the sub-formal context based on each individual entity (object or attribute).
In this way, the information intrinsic to each label can be clearly expressed.
At the same time, it reduces time consumption.
Regarding classification, we define label confidence through fusing object and attribute induced concept sets.
Respective calculation does not require additional prediction methods, thus becoming more efficient.
Experiments are conducted on fifteen public datasets from UCI and KEEL in comparison with eight classical classification algorithms.
Results validate the time complexity analysis, and show competitive classification performance of our algorithm.

Keywords:
	Classification 
 
	Formal concept analysis 
 
	Label concept 
 
	Label confidence
 
 ------------------------------------------------------------------------------------
This project demonstrates the use of formal concept analysis to achieve classification tasks.

The main process is divided into the following parts:

(1) original datasets are discretized into binary datasets.

(2) Obtaining concept sets from binary datasets.

(3) Calculating confidence.

(4) Classification and prediction.

Please feel free to contact me (xiaozeng658@gmail.com), if you have any problem about this programme.


