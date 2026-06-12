"""ECM remodeling simulation analysis pipeline.

A publication-oriented toolkit that turns a time series of cell + ECM-fiber
geometry into quantitative structural metrics, publication figures, and
ParaView-ready 3D output.

The synthetic generator (`synthetic`) is a stand-in so the whole pipeline runs
end-to-end with zero external dependencies. Replace it with a parser of real
PhysiCell + PhysiMeSS output and the rest of the pipeline is unchanged: every
downstream module consumes the same plain `cells_*.csv` / `fibers_*.csv` files.
"""

__all__ = ["synthetic", "metrics", "export_vtk", "figures"]
