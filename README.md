# Accelerating SwiGLU with Custom Triton Kernel Fusion
* Context: SwiGLU activation in large model contains %of paramters for LLMs. 
* Challenge: Reduce the memory consumption for SwiGLU layers, without deminishing the training or inference speed. 
* Solution: Developed a custom fused Triton kernel that optimized the kernal fusion to reduce GPU memory consumption, specifically reduced the parameters saved in forward process for performing backpropagation and perform inplace parameter saving. 
* Result: Achieved 32% memory reduction at sequence length 16000, while maintain the forward calculation speed and accelerate backward calculation speed by 3.3%.

<img src="swiglu_memory_full.png" alt="Description of Image" width="500" height="300">
<img src="swiglu_speed_backward.png" alt="Description of Image" width="500" height="300">
<img src="swiglu_speed_forward.png" alt="Description of Image" width="500" height="300">

![benchmark image](swiglu_memory_full.png)
![benchmark image](swiglu_speed_backward.png)
![benchmark image](swiglu_speed_forward.png)






