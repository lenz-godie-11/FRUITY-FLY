# Dataset attribution

This project uses **MaleCNS v1.0**, produced by FlyEM (HHMI Janelia),
the University of Cambridge Department of Zoology, the MRC Laboratory of
Molecular Biology, Google Research, and their collaborators.

- Official download: https://male-cns.janelia.org/download/
- Project: https://www.janelia.org/project-team/flyem/male-cns-connectome
- Publication: *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system* (Cell, 2026).
- Data license: Creative Commons Attribution 4.0 International (CC BY 4.0):
  https://creativecommons.org/licenses/by/4.0/
- Legal text: https://creativecommons.org/licenses/by/4.0/legalcode

The downloaded data are not authored by this project. Derived artifacts preserve
original neuron IDs and source-file SHA-256 hashes. Modifications: select a
4,000- and 8,000-neuron induced subgraphs, omit autapses, normalize contact counts, assign
approximate transmitter signs, convert soma coordinates from 8-nm voxels to µm,
and learn bounded connection gains for an engineered rover task.

Neuron dynamics, sensory encoding, learning, rover control, and visualizations
are our engineering choices, not properties measured by the source project.
No endorsement by the dataset authors is implied.
