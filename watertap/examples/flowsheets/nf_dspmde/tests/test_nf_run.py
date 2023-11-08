import numpy as np
from math import log
import idaes.logger as idaeslog
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    value,
    Var,
    units as pyunits,
    assert_optimal_termination,
    TransformationFactory,
)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from pyomo.network import Port
from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    MomentumBalanceType,
    ControlVolume0DBlock,
)
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
    MCASStateBlock,
)
from watertap.unit_models.nanofiltration_DSPMDE_0D import (
    NanofiltrationDSPMDE0D,
    MassTransferCoefficient,
    ConcentrationPolarizationType,
)
from watertap.core.util.initialization import check_dof

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.testing import initialization_tester
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    badly_scaled_var_generator,
)
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import (
    Feed,
)
import idaes.logger as idaeslog

# Get default solver for testing
solver = get_solver()

m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = MCASParameterBlock(
    solute_list=["Na_+", "Cl_-"],
    diffusivity_data={("Liq", "Na_+"): 1.33e-09, ("Liq", "Cl_-"): 2.03e-09},
    mw_data={"H2O": 0.018, "Na_+": 0.023, "Cl_-": 0.035},
    stokes_radius_data={"Cl_-": 1.21e-10, "Na_+": 1.84e-10},
    charge={"Na_+": 1, "Cl_-": -1},
    activity_coefficient_model=ActivityCoefficientModel.davies,
    density_calculation=DensityCalculation.constant,
)

m.fs.unit = NanofiltrationDSPMDE0D(property_package=m.fs.properties)

m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(0.429868)
m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(0.429868)
m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(47.356)

# Fix other inlet state variables
m.fs.unit.inlet.temperature[0].fix(298.15)
m.fs.unit.inlet.pressure[0].fix(4e5)

# Fix the membrane variables that are usually fixed for the DSPM-DE model
m.fs.unit.radius_pore.fix(0.5e-9)
m.fs.unit.membrane_thickness_effective.fix(1.33e-6)
m.fs.unit.membrane_charge_density.fix(-27)
m.fs.unit.dielectric_constant_pore.fix(41.3)

# Fix final permeate pressure to be ~atmospheric
m.fs.unit.mixed_permeate[0].pressure.fix(101325)

m.fs.unit.spacer_porosity.fix(0.85)
m.fs.unit.channel_height.fix(5e-4)
m.fs.unit.velocity[0, 0].fix(0.25)
m.fs.unit.area.fix(50)
# Fix additional variables for calculating mass transfer coefficient with spiral wound correlation
m.fs.unit.spacer_mixing_efficiency.fix()
m.fs.unit.spacer_mixing_length.fix()

check_dof(m, fail_flag=True)

m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e2, index=("Liq", "Cl_-")
)
m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e2, index=("Liq", "Na_+")
)
m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e0, index=("Liq", "H2O")
)

calculate_scaling_factors(m)

# check that all variables have scaling factors
unscaled_var_list = list(unscaled_variables_generator(m.fs.unit))
assert len(unscaled_var_list) == 0

# Expect only flux_mol_phase_comp to be poorly scaled, as we have not
# calculated correct values just yet.
for var in list(badly_scaled_var_generator(m.fs.unit)):
    assert "flux_mol_phase_comp" in var[0].name

initialization_tester(m)

results = solver.solve(m)

# Check for optimal solution
assert_optimal_termination(results)

pressure_steps = np.linspace(1.8e5, 20e5, 10)

for p in pressure_steps:
    m.fs.unit.inlet.pressure[0].fix(p)
    results = solver.solve(m)
    assert_optimal_termination(results)

m.fs.unit.inlet.pressure[0].fix(10e5)
m.fs.unit.area.unfix()

for r in np.linspace(0.05, 0.97, 10):
    m.fs.unit.recovery_vol_phase.fix(r)
    print(r)
    res = solver.solve(m, tee=True)
    assert_optimal_termination(res)