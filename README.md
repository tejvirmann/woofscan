# vetscan
Machine learning to look at vet scans from pets and diagnose them based on a large dataset. 

This is still a work in progress. I tested using human lung x-rays first, and it worked pretty well, but now I need to find x-rays for pets, and if you have a good way to find them let me know. I am also 
trying to develope a online web interface to people can access it online. I'll get there, lmk if you have any questions. 

I'll write an indepth explanation for how the software works when I am done, this is mostly here because my computer does not have enough space. 

In simple terms, I use the AI that was mostly trained by google, then I leverage that software, and remove the 'top layer' of what it is learned, then I 
trained to figure out if a lung had Pneumonia or if it did not. That is it so far. 

