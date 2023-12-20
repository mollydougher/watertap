# adapted from https://watertap.readthedocs.io/en/stable/how_to_guides/how_to_setup_simple_RO.html
# simple RO example

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    maximize,
    units as pyunits,
    assert_optimal_termination,
    TransformationFactory,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors
from watertap.property_models.LiCl_prop_pack import LiClParameterBlock
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
import idaes.core.util.scaling as iscale
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed,Product


def main():
    solver = get_solver()
    m = build()

    initialize(m, solver)
    print("init_okay")
    m.fs.unit.report()

    assert degrees_of_freedom(m) == 0
    optimize(m, solver)
    print("solved box problem")
    m.fs.unit.report()

    unfix_opt_vars(m)
    add_obj(m)
    # add_con(m)
    optimize(m, solver)
    m.fs.unit.report()

    return m


def set_default_feed(blk):
    # mass flow rate (kg/s)
    blk.feed.flow_mass_phase_comp[0, 'Liq', 'LiCl'].fix(0.03158)
        # increasing the Li ion concentration fixed the permeate initializtion fail
        # currently 10x too high for this ratio of water
    blk.feed.flow_mass_phase_comp[0, 'Liq', 'H2O'].fix(1.058)

    # Set scaling factors for component mass flowrates.
    blk.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    blk.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'LiCl'))


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = LiClParameterBlock()

    # add the feed and product streams
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.permeate = Product(property_package=m.fs.properties)
    m.fs.retentate = Product(property_package=m.fs.properties)

    # define unit models
    m.fs.unit = ReverseOsmosis0D(
        property_package=m.fs.properties,
        concentration_polarization_type=ConcentrationPolarizationType.none,
        mass_transfer_coefficient=MassTransferCoefficient.none,
        has_pressure_change=False)
    
    # connect the streams and blocks
    m.fs.feed_to_ro = Arc(source=m.fs.feed.outlet, destination=m.fs.unit.inlet)
    m.fs.ro_to_permeate = Arc(source=m.fs.unit.permeate, destination=m.fs.permeate.inlet)
    m.fs.ro_to_retentate = Arc(source=m.fs.unit.retentate, destination=m.fs.retentate.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    return(m)


def fix_init_vars(m):
    # feed pressure (Pa)
    m.fs.unit.inlet.pressure[0].fix(5e5)
    # feed temperature (K)
    m.fs.unit.inlet.temperature[0].fix(298.15)
    # membrane area (m^2)
    m.fs.unit.area.fix(50)
    # membrane water permeability (m/Pa/s)
    m.fs.unit.A_comp.fix(4.2e-12)
    # membrane salt permeability (m/s) # TODO: verify
    m.fs.unit.B_comp.fix(3.5e-8)
    # permeate pressure (Pa)
    m.fs.unit.permeate.pressure[0].fix(101325)


def unfix_opt_vars(m):
    m.fs.unit.area.unfix()


def add_obj(m):
    # min specific energy consumption
    m.fs.obj = Objective(
        expr = m.fs.unit.inlet.pressure[0]/(3.6e6),
        # sense = maximize
    )


def optimize(m, solver):
    print("Optimizing with {} DOFs".format(degrees_of_freedom(m)))
    simulation_results = solver.solve(m, tee=True)
    assert_optimal_termination(simulation_results)
    return simulation_results


def initialize(m, solver):
    set_default_feed(m.fs)
    fix_init_vars(m)

    m.fs.feed.initialize(optarg=solver.options)
    propagate_state(m.fs.feed_to_ro)

    m.fs.unit.initialize(optarg=solver.options)
    propagate_state(m.fs.ro_to_permeate)
    propagate_state(m.fs.ro_to_retentate)

    m.fs.permeate.initialize(optarg=solver.options)
    m.fs.retentate.initialize(optarg=solver.options)


if __name__ == "__main__":
    main()
