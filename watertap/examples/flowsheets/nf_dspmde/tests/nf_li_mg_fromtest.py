####################################################################
# This code has been adapted from a test in the WaterTAP repo to
# simulate a simple separation of lithium and magnesium
#
# watertap > unit_models > tests > test_nanofiltration_DSPMDE_0D.py
# test defined as test_pressure_recovery_step_2_ions()
#
# also used the following flowsheet as a reference
# watertap > examples > flowsheets > nf_dspmde > nf.py
#
# https://github.com/watertap-org/watertap/blob/main/tutorials/nawi_spring_meeting2023.ipynb
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

from watertap.unit_models.pressure_changer import Pump

from watertap.unit_models.nanofiltration_DSPMDE_0D import (
    NanofiltrationDSPMDE0D,
    MassTransferCoefficient,
    ConcentrationPolarizationType,
)
from watertap.core.util.initialization import check_dof
from idaes.core.util.model_statistics import degrees_of_freedom

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
    Product
)

import numpy as np


# main function from Alex Dudchenko
def main():
    # solver = get_solver()
    m = build()
    m.fs.pump.initialize()
    m.fs.unit.initialize()
    print("init_okay")
    m.fs.unit.report()
    assert degrees_of_freedom(m) == 0
    # results = solver.solve(m)
    # assert_optimal_termination(results)
    return m


def feed_properties():
    property_kwds = {
        # need to add Cl- for electroneutrality, assume LiCl and MgCl2 salts
        "solute_list": ["Li_+","Mg_2+","Cl_-"],
        # https://www.aqion.de/site/diffusion-coefficients
        # very confident
        "diffusivity_data": {
            ("Liq","Li_+"): 1.03e-09,
            ("Liq","Mg_2+"): 0.705e-09,
            ("Liq","Cl_-"): 2.03e-09
        },
        # very confident
        "mw_data": {
            "H2O": 0.018,
            "Li_+": 0.0069,
            "Mg_2+": 0.024,
            "Cl_-": 0.035
        },
        # avg vals from https://www.sciencedirect.com/science/article/pii/S138358661100637X
        # medium confident, these values come from above review paper, averaged values from multiple studies
        # reasonable orders of magnitude
        "stokes_radius_data": {
            "Cl_-": 0.121e-9,
            "Li_+": 3.61e-10,
            "Mg_2+": 4.07e-10,
            #"Cl_-": 3.28e-10
        },
        # very confident
        "charge": {
            "Li_+": 1,
            "Mg_2+": 2,
            "Cl_-": -1
        },
        # choose ideal for now, other option is davies
        "activity_coeficcient_model":ActivityCoefficientModel.ideal,
        "density_calculation": DensityCalculation.constant
    }


def build():
    # create the model
    m = ConcreteModel()

    # create the flowsheet
    m.fs = FlowsheetBlock(dynamic = False)

    # define the propery model
    m.fs.properties = MCASParameterBlock(**feed_properties)

    # add the feed and product streams
    m.fs.feed = Feed(property_package = m.fs.properties)
    m.fs.product = Product(property_package = m.fs.properties)
    m.fs.disposal = Product(property_package = m.fs.properties)

    # define unit models
    m.fs.pump = Pump(property_package = m.fs.properties)
    m.fs.unit = NanofiltrationDSPMDE0D(property_package = m.fs.properties)

    # connect the streams and blocks
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_nf = Arc(source=m.fs.pump.outlet, destination=m.fs.unit.inlet)
    m.fs.nf_to_product = Arc(source=m.fs.unit.permeate, destination=m.fs.product.inlet)
    m.fs.nf_to_disposal = Arc(source=m.fs.unit.retentate, destination=m.fs.disposal.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # fix the inlet flow rates
    # approximate the values of Salar de Atacama (Cl- overridden)
    m.fs.unit.feed_side.properties_in[0.0].flow_mol_phase_comp["Liq", "Li_+"].fix(0.172)
    m.fs.unit.feed_side.properties_in[0.0].flow_mol_phase_comp["Liq", "Mg_2+"].fix(0.305)
    m.fs.unit.feed_side.properties_in[0.0].flow_mol_phase_comp["Liq", "Cl_-"].fix(0.42)
    m.fs.unit.feed_side.properties_in[0.0].flow_mol_phase_comp["Liq", "H2O"].fix(47)

    # assert electroneutrality
    m.fs.unit.feed_side.properties_in[0.0].assert_electroneutrality(
        defined_state = True,
        adjust_by_ion = "Cl_-",
        get_property = "flow_mol_phase_comp"
    )

    # fix the inlet fstate variables
    m.fs.unit.inlet.temperature[0].fix(298.15)
    m.fs.unit.inlet.pressure[0].fix(2e5)

    # fix the membrane properties, typical for DSPM-DE
    # kept all values from the WaterTAP test.py for now
    m.fs.unit.radius_pore.fix(0.5e-9)
    m.fs.unit.membrane_thickness_effective.fix(1.33e-6)
    m.fs.unit.membrane_charge_density.fix(-60)
    m.fs.unit.dielectric_constant_pore.fix(41.3)

    # fix the pump variables
    m.fs.pump.efficiency_pump[0].fix(0.75)
    m.fs.pump.outlet.pressure[0].fix(2e5)

    # fix final permeate pressure to be approx atmospheric
    m.fs.unit.mixed_permeate[0].pressure.fix(101325)

    # fix system values
    m.fs.unit.spacer_porosity.fix(0.85)
    m.fs.unit.channel_height.fix(5e-4)
    m.fs.unit.velocity[0, 0].fix(0.1)
    m.fs.unit.area.fix(100)

    # fix additional variables for calculating mass transfer coefficient with spiral wound correlation
    m.fs.unit.spacer_mixing_efficiency.fix()
    m.fs.unit.spacer_mixing_length.fix()

    # check the DOF
    check_dof(m, fail_flag = True)

    # scaling, using same method as WaterTAP test.py for now
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1 / 0.5, index = ("Liq", "Li_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1 / 0.5, index = ("Liq", "Mg_2+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1, index = ("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1 / 47, index = ("Liq", "H2O")
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
    return m


if __name__ == "__main__":
    main()
