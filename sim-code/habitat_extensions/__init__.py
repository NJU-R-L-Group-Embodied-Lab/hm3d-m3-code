"""habitat_extensions: task, simulator, sensor and measure extensions for Habitat.

Derived from the VLN-CE ``habitat_extensions`` package
(https://github.com/jacobkrantz/VLN-CE) and adapted for HM3D-M3 multi-agent
ObjectNav on Habitat v0.1.7. Importing this package registers the custom
datasets, simulators, sensors and measures with the Habitat registry, so the
imports below are intentional side effects.
"""
from habitat_extensions.task import VLNCEDatasetV1
from habitat_extensions.habitat_simulator import Simulator
from habitat_extensions import measures
from habitat_extensions import sensors