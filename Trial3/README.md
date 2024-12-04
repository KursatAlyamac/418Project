# SuperTuxKart Trial 3

## Planner Settings
- **Model**: 
    - `CNN with 5 layers`
    - `Utilizes MaxPool and AvgPool`
    - `Utilizes Dropout layers`
    - `Implements a regression head`
- **Learning Rate**: `1e-4`  
- **Epochs**: `300`  
- **Dataset**: `30,000 images`  

## Results
- **Loss**: `0.042`  
 
## Course times (Best runs)
- **Zengarden**: `t = 444`  
- **Lighthouse**: `t = 492`  
- **Hacienda**: `t = 570`  
- **Snowtuxpeak**: `t = 823`  
- **Cornfield Crossing**: `t = 793`  
- **Scotland**: `t = 669`  

---

## Summary
- Increased images to see if large CNN just needed larger dataset
- Struggled about the same as previous one, seems to be a problem with cnn and not dataset, extremely struggled with snowtux
- Revert CNN next trial, test old CNN with large dataset

