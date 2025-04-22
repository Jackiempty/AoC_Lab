# Lab 4 Homework Template

## PASS Screenshot
### DLA (total 30%)
##### Testbench `dla0` 7.5%

##### Testbench `dla1` 7.5%

##### Testbench `dla2` 7.5%

##### Testbench `dla3` 7.5%


### CPU (total 60%)
Each `original version` operation contributes **7.5** points to the overall score.
Each `Improved version` operation is based on **the degree of cycle reduction**, with a full score of 7.5 points.
Please refer to the section below for detailed evaluation criteria.
##### `qconv2d_relu_maxpool_cpu`
**Original**

**Improve**

##### `qconv2d_relu_cpu`
**Original**

**Improve**

##### `qlinear_relu_cpu`
**Original**

**Improve**

##### `qlinear_cpu`
**Original**

**Improve**

## Performance in CPU
Please fill in the number of execution cycles and the results from the Valgrind analysis.
| Function     |            D refs      |        D1 miss      |   Cycle (roughly), depend on server status  |
|------------------|--------------------|---------------------|-----------|
|`CONV_original`         |         |            |          |
|`CONV_improve`          |         |            |          |
|`CONV_MAX_original`     |         |            |          |
|`CONV_MAX_improve`      |         |            |          |
|`LINEAR_original`       |         |            |          |
|`LINEAR_improve`        |         |            |          |
|`LINEAR_RELU_original`  |         |            |          |
|`LINEAR_RELU_improve`   |         |            |          |

**Scoring Criteria for every improved operation:**

| Cycles reduced ratio  | score | Note                  |
| -------------------------------- | ----- | --------------------- |
| <20%                             | 2.5%   | Basic implement score |
| 20%                              | 3.5%   |                   |
| 30%                              | 5%   |                   |
| 40%                              | 6.5%   |                   |
| 50%                              | 7.5%   |                   |

Cycle Reduction Ratio, defined as:
$$
\frac{\text{Cycle}_{\text{original}} - \text{Cycle}_{\text{improved}}}{\text{Cycle}_{\text{original}}}
$$

:::info
**❗Additional information:**
Since the number of D refs and D1 misses does not vary with the server's state, these values will be used as a reference during grading. The cycle count, however, will be evaluated with a more lenient standard.
:::


## How to improve performance in CPU 10%
:::info
Explain your method.
:::


## Feedback bonus 10%
