# adapted from https://watertap.readthedocs.io/en/stable/how_to_guides/how_to_setup_simple_RO.html
# simple RO example

from pyomo.environ import ConcreteModel, assert_optimal_termination
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors
from watertap.property_models.LiCl_prop_pack import LiClParameterBlock
from watertap.unit_models.reverse_osmosis_0D import ReverseOsmosis0D
from watertap.unit_models.reverse_osmosis_0D import ConcentrationPolarizationType
from watertap.unit_models.reverse_osmosis_0D import MassTransferCoefficient
from idaes.core.solvers import get_solver


# Create a concrete model, flowsheet, and LiCl property parameter block.
m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = LiClParameterBlock()

# Add an RO unit to the flowsheet.
m.fs.unit = ReverseOsmosis0D(
    property_package=m.fs.properties,
    concentration_polarization_type=ConcentrationPolarizationType.none,
    mass_transfer_coefficient=MassTransferCoefficient.none,
    has_pressure_change=False)

# Specify system variables.
m.fs.unit.inlet.flow_mass_phase_comp[0, 'Liq', 'LiCl'].fix(0.03158)  # mass flow rate of LiCl (kg/s)
        # increasing the Li ion concentration fixed the permeate initializtion fail
        # currently 10x too high for this ratio of water
m.fs.unit.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'].fix(1.058)   # mass flow rate of water (kg/s)
m.fs.unit.inlet.pressure[0].fix(5e5)                              # feed pressure (Pa)
m.fs.unit.inlet.temperature[0].fix(298.15)                         # feed temperature (K)
m.fs.unit.area.fix(50)                                             # membrane area (m^2)
m.fs.unit.A_comp.fix(4.2e-12)                                      # membrane water permeability (m/Pa/s)
m.fs.unit.B_comp.fix(3.5e-8)                                       # membrane salt permeability (m/s)
m.fs.unit.permeate.pressure[0].fix(101325)                         # permeate pressure (Pa)

# Set scaling factors for component mass flowrates.
m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'LiCl'))

# Calculate scaling factors.
calculate_scaling_factors(m)

# Initialize the model.
m.fs.unit.initialize()

# solve the model
solver = get_solver()
simulation_results = solver.solve(m, tee=True)
assert_optimal_termination(simulation_results)