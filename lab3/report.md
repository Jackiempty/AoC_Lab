# Lab 3 Report

## Lab Design Description (40%)
Scores will be assigned based on the level of detail and the logical soundness of your description.
Each section should ideally include a block diagram to explain the module.

### Explain how you implement PE
> Explain how you handle the computation of the zero point of dequantization.
### Explain how you implement PE array
> Including network (GIN/GON/LN) and multicast controller (MC)

### Explain how you implement PPU
> Explain how you handle the computation of the zero point of requantization \
> Explain the order of 3 component(PostQuant, MaxPool, ReLU) and the reason why you design like that.

### Result

| Component | Pass (Y/N) |
|:---------:|:----------:|
|    PE     |            |
| PE array  |            |
|    PPU    |            |


## Question (50%)

### Question 1 (20%)

Explain how data reuse is achieved in the design presented in the Eyeriss paper.

### Question 2 (10%)

Compute a `16×16` Conv2D operation, given the following configuration
- Kernel size: `3×3`
- Stride: `1`
- Padding: `1`
- Global Buffer (GLB) size: `128 KB`
- Mapping parameters:
    - `p = 4`
    - `q = 4`
    - `r = 1`
    - `t = 2`
    - `e = 8`

Determine the value of the mapping parameter `m`

**Please include the calculation process.**

### Question3 (10%)
In the testbench, explain how to compute the multicast controller’s ID (`X_ID`, `Y_ID`) and the data tags (`tag_X`, `tag_Y`) based on the mapping parameters and shape parameters.

### Question 4 (10%)

For the test case where e = 4,
why are the configurations `(r, t)` = `(1, 4)` and `(2, 2)` present, but `(4, 1)` is not?
What do you think is the reason behind this?

## Lab Feedback and Suggestions (10%)