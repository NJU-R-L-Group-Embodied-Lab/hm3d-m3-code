"""HM3D-M3 benchmark — episode generation pipeline.

P1 = the geometry "foundation": multi-agent distributed starts with joint
reachability verification + difficulty-stratified sampling, producing a
stratified small set. IIN image rendering and TN description generation are
deliberately deferred to P2; here we only *tag* goal modalities and record the
viewpoint each goal can be rendered from.

All P1 geometry runs from the HM3D ``.navmesh`` files via ``habitat_sim.PathFinder``
— no Simulator, no GPU, no sensors — see :class:`benchmark.generate.HabitatNav`.
"""
