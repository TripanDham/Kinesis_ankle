Baseline: 
- Kinesis
- Using an able bodied musculoskeletal model - 80 muscles
- Trained on human able bodied walking data - for walking at various speeds, turning, sideways, circle walking, etc. 
- Proven to have human like EMG. 
- Shows poor performance (upwards of 8 cm error) with data from amputees walking

Attempt 1: 
- Imitation learning with simple Minimax discriminator
- discriminator figures agent is cheating very easily, very bad reward signals
- added motor penalty, changed motor limits, reduced friction
- agent did not learn to walk, learnt to tap and stay alive

Attempt 2: 
- WGAN discriminator

Attempt 3:
- included RSI - maybe the agent does not have enough information to learn how to start walking

Attempt 3:
- reference tracking instead of imitation learning
- joint tracking
- local feature learnt amazingly, but then agent doesnt walk

Attempt 4:
- reference tracking but with body positions


Data:
- 18 amputees walking - variety of left and right, and some walking with handrails and some without. 
- 13 seconds of walking each, walk at different speeds too
- parsed using opensim - using Gait 2932 model, with added markers for calibration - rigid triangles on thigh and prosthetic leg. 
- Issues seen: prosthetic marker placement is very different from human marker placement so this led to errors in the feet and so on. 
- The joint angles are replicated in mujoco
- since all subjects are of different height this was a slight issue - needed scaling for individual subjects which needed manual tweaking.
- the artefact due to the shoe below the prosthesis caused issues with the foot touching the ground in mujoco - and hence a small tweak to make the legs a little imbalanced was useful. 
- velocities are computed using finite differences
