# SuperTuxKart Project

Welcome to our SuperTuxKart project repository! This repository contains all the code and experiments we have worked on to achieve the best possible results in SuperTuxKart. Each folder represents a specific phase or focus area in the development process, with names reflecting the progress made.

## Repository Structure

### `default-codebase`
Contains the original codebase provided with our custom controller implementation.

### `early_trials`
Initial experiments and developments, including:
- Adjustments to image inputs
- Modifications to CNN layers
- Changes to the controller logic
- Hyperparameter tuning

### `modified_cnns`
Enhancements made to the CNN architecture, such as:
- Adding dropout layers
- Introducing pooling layers
- Other architectural modifications

### `advanced_pathing`
Focused on creating optimal pathing strategies for the kart.

### `powerups_versions`
Features implemented in this folder include:
- Obstacle detection
- Racing line prediction
- Enhanced neural network functionalities

### `cart_ddpg`
Reinforcement learning implementation for controller acceleration using Double Q-Learning (DDPG).

### `best_results`
Contains the overall best-performing models and configurations:
- Two trials are included: one with more epochs and another with fewer epochs.
- Combines features from `powerups_versions`, `modified_cnns`, and `advanced_pathing` for optimal performance.

---

Feel free to explore the folders and review the progression of our work. Each directory provides insight into the iterations and improvements we have made throughout the project.
