# Accelerating SwiGLU with Custom Triton Kernel Fusion
* Context: SwiGLU activation in large model training/inference suffers from significant I/O overhead. 
* Challenge: Improve computation speed of the Feed Forward Layer (FFD) by reducing intermediate memory traffic. 
* Solution: Developed a custom fused Triton kernel to merge multiple operations, minimizing I/O and improving execution efficiency. 
* Result: Achieved 32% memory reduction at sequence length 16000, while maintain the forward calculation speed and accelerate backward calculation speed by 3.3%.

![benchmark image](swiglu_memory_full.png)
![benchmark image](swiglu_speed_backward.png)
![benchmark image](swiglu_speed_forward.png)
