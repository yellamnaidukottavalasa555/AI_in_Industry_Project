# Improve usage of unsupervised data for the definition of RUL-based maintenance policies
This is the final project for the Artificial Intelligence in Industry course at UniBo.

In the recent years, industries such as aeronautical, railway, and petroleum has transitioned from corrective or preventive maintenance to condition based maintenance (CBM). One of the enablers of CBM is Prognostics which primarily deals with prediction of remaining useful life (RUL) of an engineering asset. Besides physics-based approaches, data driven methods are widely used for prognostics purposes, however the latter technique requires availability of run to failure datasets. In this project we show a method to inject external knowledge into the deep learning model to exploit data about the normal functioning of the machines. We apply our approach on the Commercial Modular Aero-Propulsion System Simulation (CMAPSS) model developed at NASA.

## Project Work Flow

### Dataset

Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which was developedby NASA. The CMAPSS dataset includes 4 sub-datasets that are composed of multi-variate temporal data obtained from 21 sensors. Each sub-dataset contains one training set and one test set. The training datasets include run-to-failure sensor records of multiple aero-engines collected under different operational conditions and fault modes.

| Dataset               | FD001        | FD002 | FD003        | FD004    | 
| :--------------------:| :-----------:| :----:| :-----------:| :-------:|
| Training Trajectories | 100          | 260   | 100          | 248      |
| Test Trajectories     | 100          | 259   | 100          | 249      |
| Operating Conditions  | 1(sea level) | 6     | 1(sea level) | 6        |
| Fault Modes           | HPC          | HPC   |  HPC, Fan    | HPC, Fan |

### Dataset generation

To simulate the scarcity of the data at various level, we define a series of ratios in which the data set will be splitted in supervised training set, unsupervised training set and test set. The partitions are defined by the mean of the machines, in this way we maintain together the sample about the same machine. The size of the test set is fixed to 12% for all the experiments. The below table shows how we have partitioned the data set.

|     %    | no.of supervised machine/samples | no.of unsupervised machine/samples | machine/samples |
| :-------:| :-------------------------------:| :---------------------------------:| :--------------:|
| 3%/ 75%  |              7/1548              |              186/45470             |     56/14231    |
| 23%/ 55% |              57/13367            |              136/33651             |     56/14231    |
| 43%/ 35% |              107/25853           |              87/21505              |     55/13891    |
| 63%/ 15% |              156/38215           |              37/8803               |     56/14231    |

### Experiments

**four models**

* First model is the baseline model.

* Second model applies the semantic-based regularizer with a fixed constraint equal to 1.

* Third model applies both the semantic-based regulazer and the penalty term is chosen using the Lagrangian Dual Framework. It has a single penalty term computed on the average of all the constraint.

* Fourth model applies both the approaches but it has multiple penalty term.

### Conclusion

In this project we have applied a semi-supervised approach for estimating the Remain Useful Life. The approach looks promising when used with a small quantity of supervised data. The problem is the high average of remaining days to the end-of-life for each machine, predicted by the model on which we have applied the regularizer. This is probably due to the too strictly satisfaction of the constraints but it needs further investigation.

## Authors
Davide Angelani

Yellam Naidu Kottavalasa

Usha Padma Rachakonda
