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
    diffusivity_data = {
        ("Liq","Li_+"): 1.03e-09,
        ("Liq","Mg_2+"): 0.075e-09
    },
    mw_data = {
        "H2O": 0.018,
        "Li_+": 0.0069,
        "Mg_2+": 0.024
    },
    # avg vals from https://www.sciencedirect.com/science/article/pii/S138358661100637X
    stokes_radius_data = {
        "Li_+": 3.61e-10,
        "Mg_2+": 4.07e-10
    },
    charge = {
        "Li_+": 1,
        "Mg_2+": 2
    },
    # choose ideal for now, other option is davies
    activity_coefficient_model=ActivityCoefficientModel.ideal,
    density_calculation=DensityCalculation.constant
)

# define the unit model
