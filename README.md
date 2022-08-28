# Knowledge Distillation on Custom Architectures
This repository contains implementation of online, offline and self-distillation methods

-   Offline Distillation
    -   TAKD: Improved Knowledge Distillation via Teacher Assistant.

-   Online Distillation:
    -   RCO: Knowledge Distillation via Route Constrained Optimization.

-   Self-Knowledge Distillation:
    -   CSKD: Regularizing class-wise predictions via self-knowledge distillation.
    -   Self-Training: Revisit Knowledge Distillation – A Teacher-free framework.


## Baseline Accuracies
### Custom and Standard Architectures
#### DW - Depth-wise separable convolutions

| No | Model            | Trainable Parameters | Params size (MB) | Estimated Total Size (MB) | "Baseline accuracy (%) CIFAR-10" | "Baseline Accuracy (%) FMNIST" |
|----|------------------|----------------------|------------------|---------------------------|----------------------------------|--------------------------------|
| 1  | model_1M_wo_dw   | 1,180,970            | 4.51             | 8.26                      | 87.8                             | 92.67                          |
| 2  | model_1M_w_dw    | 1,159,474            | 4.42             | 16.64                     | 87.95                            | 91.89                          |
| 3  | model_600k_wo_dw | 590,378              | 2.25             | 5.64                      | 88.45                            | 92.93                          |
| 4  | model_600k_w_dw  | 599,913              | 2.29             | 13.41                     | 88.71                            | 92.56                          |
| 5  | model_340k_wo_dw | 340,010              | 1.30             | 3.86                      | 87.7                             | 93.17                          |
| 6  | model_340k_w_dw  | 344,508              | 1.31             | 4.65                      | 85.79                            | 90.28                          |
| 7  | model_143k_wo_dw | 143,218              | 0.55             | 2.06                      | 83.59                            | 92.68                          |
| 8  | model_143k_w_dw  | 143,406              | 0.55             | 2.48                      | 83.93                            | 90                             |
| 9  | model_25k_wo_dw  | 25,298               | 0.10             | 0.99                      | 76.54                            | 90.7                           |
| 10 | model_25k_w_dw   | 25,612               | 0.10             | 0.95                      | 72.58                            | 90.27                          |
| 11 | ResNet-18        | 11,181,642           | 42.65            | 43.95                     | 82.59                            | 89.06                          |
| 12 | ResNet-34        | 21,289,802           | 81.21            | 83.19                     | 84.44                            | 87.18                          |
| 13 | ResNet-50        | 23,528,522           | 89.75            | 95.63                     | 82.47                            | 90.21                          |
| 14 | ResNet-101       | 42,520,650           | 162.20           | 171                       | 80.73                            | 88.47                          |
| 15 | ResNet-152       | 58,164,298           | 221.88           | 234.29                    | 79.82                            | 88.58                          |
| 16 | EfficientNet B5  | 28,361,274           | 108.19           | 130.08                    | 89.59                            | 89.78                          |
| 17 | EfficientNet B7  | 63,812,570           | 243.43           | 281.5                     | 91.08                            | 92.01                          |
