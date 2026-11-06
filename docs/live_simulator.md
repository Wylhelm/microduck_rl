# Live body simulator: visual mouth

`duck-body` renders the commanded mouth for each duck in the passive MuJoCo
viewer and in enabled simulated camera frames. It uses the same render-only
world for both. The runtime sends targets in radians at wire index 9:
-5 degrees is closed and +30 degrees fully open; the display clamps to this
range. Coin-Coin's audio-synchronized mouth commands therefore animate the
beak during speech. A closed target closes it; torque-off and daemon disconnect
also close the display. Without a closing command, the last target remains
visible while the connection stays open.

The render-only copy rotates the visual jaw meshes around an approximate
hinge; the motion is illustrative, not a calibrated servo or linkage.
Physics, collision meshes, mouth-tip site, inertia, joints, actuators,
observations and the 14-action / 61D policy contract are unchanged. Mouth
position, velocity and current in simulated sensor replies remain zero: this
cannot validate a physical mouth servo or its feedback.

The default model and attached ducks supply the required jaw meshes. A custom
scene lacking either visual mesh logs a warning and has no animated beak.
The head camera is mounted on the duck, so it may not see its own beak; other
ducks' beaks can appear in its view. `--headless` skips the window, while
`--cameras a` (or `--cameras all`) separately opts into camera rendering.
