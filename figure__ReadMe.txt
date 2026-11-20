The figures plotted for the paper are generated with following scripts.
Scripts create *.svg, *.png and *.pdf.

In the order of existance within the paper:
figure_NonStructuredModel.py:       Plots Non-structured DNN weight and neuron pruning

figureNonStructuredMatrix.py        Plots Non-structured pruning weight matrix

figure_DnnAccelerators.py:          Plot DNN Accelerators (ASIC, FPGA, NoC, CPU/GPU) figure

figureNoCtopologies.py:             Plot NoC topologies: 2d mesh, torus, condensed mesh

figure_GroupWctgPEmapExample.py:    Plot network showing the DNN graph with 
                                    neuron groupings, 
                                    WCTG, 
                                    and PE mappings for a small 4-6-6-1 DNN example
                                  
figure_Pipeline_overview.py:        Plot proposed 5 stage End-to-end pipeline overview.

figure_ExperimentalResults.py:      Plot experimental results:
                                    For (LeNet-300-100 / MNIST):
                                        1- Comm. cost vs. accuracy drop  
                                        2- FLOPs vs. accuracy drop
                                        3- Comm. cost proxy vs. hidden neurons remaining
                                        4- Comm. cost proxy vs. edges remaining
                                    For LeNet-500-300 / MNIST: 
                                        5- WCTG neuron to neuron comm. cost vs. accuracy budget






 