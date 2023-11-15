####################################################################
# This code has been adapted from a test in the WaterTAP repo to
# simulate a simple separation of lithium and magnesium
#
# watertap > unit_models > tests > test_nanofiltration_DSPMDE_0D.py
# test defined as test_pressure_recovery_step_2_ions()
####################################################################

# import statements
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

# define the default solver
solver = get_solver

# create the model
m = ConcreteModel()

# create the flowsheet
m.fs = FlowsheetBlock(dynamic = False)

# define the propery model
m.fs.properties = MCASParameterBlock(
    solute_list = ["Li_+", "Mg_2+"],
    # https://www.aqion.de/site/diffusion-coefficients
    # very confident
    diffusivity_data = {
        ("Liq","Li_+"): 1.03e-09,
        ("Liq","Mg_2+"): 0.075e-09
    },
    # very confident
    mw_data = {
        "H2O": 0.018,
        "Li_+": 0.0069,
        "Mg_2+": 0.024
    },
    # avg vals from https://www.sciencedirect.com/science/article/pii/S138358661100637X
    # medium confident, these values come from above review paper, averaged values from multiple studies
    # reasonable orders of magnitude
    stokes_radius_data = {
        "Li_+": 3.61e-10,
        "Mg_2+": 4.07e-10
    },
    # very confident
    charge = {
        "Li_+": 1,
        "Mg_2+": 2
    },
    # choose ideal for now, other option is davies
    activity_coefficient_model=ActivityCoefficientModel.ideal,
    density_calculation=DensityCalculation.constant
)

# define the unit model
m.fs.unit = NanofiltrationDSPMDE0D(property_package = m.fs.properties)

# fix the inlet flow rates
# kept all values from the WaterTAP test.py for now
m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Li_+"].fix(0.429868)
m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Mg_2+"].fix(0.429868)
m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(47.356)

# fix the inlet fstate variables
# kept all values from the WaterTAP test.py for now
m.fs.unit.inlet.temperature[0].fix(298.15)
m.fs.unit.inlet.pressure[0].fix(4e5)

# fix the membrane properties, typical for DSPM-DE
# kept all values from the WaterTAP test.py for now
m.fs.unit.radius_pore.fix(0.5e-9)
m.fs.unit.membrane_thickness_effective.fix(1.33e-6)
m.fs.unit.membrane_charge_density.fix(-27)
m.fs.unit.dielectric_constant_pore.fix(41.3)

# fix final permeate pressure to be approx atmospheric
m.fs.unit.mixed_permeate[0].pressure.fix(101325)

# fix system values
# kept all values from the WaterTAP test.py for now
m.fs.unit.spacer_porosity.fix(0.85)
m.fs.unit.channel_height.fix(5e-4)
m.fs.unit.velocity[0, 0].fix(0.25)
m.fs.unit.area.fix(50)

# fix additional variables for calculating mass transfer coefficient with spiral wound correlation
m.fs.unit.spacer_mixing_efficiency.fix()
m.fs.unit.spacer_mixing_length.fix()

# check the DOF
check_dof(m, fail_flag = True)

# scaling, using same method as WaterTAP test.py for now
m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e2, index = ("Liq", "Li_+")
)
m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e2, index = ("Liq", "Mg_2+")
)
m.fs.properties.set_default_scaling(
    "flow_mol_phase_comp", 1e0, index = ("Liq", "H2O")
)
# calculate the scaling factors
calculate_scaling_factors(m)
# check that all variables have scaling factors
unscaled_var_list = list(unscaled_variables_generator(m.fs.unit))
assert len(unscaled_var_list) == 0
# Expect only flux_mol_phase_comp to be poorly scaled, as we have not
# calculated correct values just yet.
for var in list(badly_scaled_var_generator(m.fs.unit)):
    assert "flux_mol_phase_comp" in var[0].name

# from idaes.core.util.model_diagnostics import DiagnosticsToolbox
# dt = DiagnosticsToolbox(m)
# dt.report_structural_issues()

# initialize model
# initialization_tester(m)

from idaes.core.util.model_diagnostics import DiagnosticsToolbox
dt = DiagnosticsToolbox(m)
dt.report_numerical_issues()

# solve model
#results = solver.solve(m)