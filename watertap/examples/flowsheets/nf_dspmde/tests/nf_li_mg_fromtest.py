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
    Objective,
    maximize,
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
    report_statistics
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

def main():
    solver = get_solver()
    m = build()
    fix_init_vars(m)
    m.fs.pump.initialize()
    m.fs.unit.initialize()
    print("init_okay")
    m.fs.unit.report()
    # add an objective, maximize Mg rejection
    m.fs.recovery_obj = Objective(
        expr = m.fs.unit.rejection_intrinsic_phase_comp[0,"Liq", "Mg_2+"],
        sense = maximize
    )
    # assert degrees_of_freedom(m) == 0
    # unfix optimization variables
    m.fs.pump.outlet.pressure[0].unfix()
    m.fs.unit.area.unfix()
    simulation_results = solver.solve(m, tee=True)
    assert_optimal_termination(simulation_results)
    m.fs.unit.report()
    report_statistics(m)
    return m


def set_default_feed(m, solver):
    # fix the feed concentrations used in the initialization
    # approximate the kg/m3 = g/L conc of Salar de Atacama (Cl- gets overridden)
    conc_mass_phase_comp = {
        "Li_+": 1.19,
        "Mg_2+": 7.31,
        "Cl_-": 143.72
    }
    set_NF_feed(
        blk=m.fs,
        solver=solver,
        flow_mass_h2o=10,   # arbitraty for now
        conc_mass_phase_comp=conc_mass_phase_comp
    )

def define_feed_comp():
    default = {
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
        # adjusted Cl and Mg to values from nf.py
        # medium confident, these values come from above review paper, averaged values from multiple studies
        # reasonable orders of magnitude
        "stokes_radius_data": {
            "Cl_-": 0.121e-9,
            "Li_+": 3.61e-10,
            "Mg_2+": 0.347e-9
            #"Mg_2+": 4.07e-10,
            #"Cl_-": 3.28e-10
        },
        # very confident
        "charge": {
            "Li_+": 1,
            "Mg_2+": 2,
            "Cl_-": -1
        },
        # choose ideal for now, other option is davies
        "activity_coefficient_model":ActivityCoefficientModel.ideal,
        "density_calculation": DensityCalculation.constant
    }
    return default


def build():
    # create the model
    m = ConcreteModel()

    # create the flowsheet
    m.fs = FlowsheetBlock(dynamic=False)

    # define the propery model
    default = define_feed_comp()
    m.fs.properties = MCASParameterBlock(**default)

    # add the feed and product streams
    m.fs.feed = Feed(property_package=m.fs.properties)
    # next 2 lines from nf.py
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.properties[0].flow_mass_phase_comp[...]
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.disposal = Product(property_package=m.fs.properties)

    # define unit models
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.unit = NanofiltrationDSPMDE0D(property_package=m.fs.properties)

    # connect the streams and blocks
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_nf = Arc(source=m.fs.pump.outlet, destination=m.fs.unit.inlet)
    m.fs.nf_to_product = Arc(source=m.fs.unit.permeate, destination=m.fs.product.inlet)
    m.fs.nf_to_disposal = Arc(source=m.fs.unit.retentate, destination=m.fs.disposal.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    
    # m.fs.unit.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "Li_+"].fix(1.19)
    # m.fs.unit.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "Mg_2+"].fix(7.31)
    # m.fs.unit.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "Cl_-"].fix(143.72)
    # m.fs.unit.feed_side.properties_in[0].flow_mol_phase_comp["Liq", "H2O"].fix(10)

    # # unfix optimization variables
    # m.fs.pump.outlet.pressure[0].unfix()
    # m.fs.unit.area.unfix()

    # check the DOF
    # check_dof(m, fail_flag = True)

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
    # unscaled_var_list = list(unscaled_variables_generator(m.fs.unit))
    # assert len(unscaled_var_list) == 0
    # Expect only flux_mol_phase_comp to be poorly scaled, as we have not
    # calculated correct values just yet.
    for var in list(badly_scaled_var_generator(m.fs.unit)):
        assert "flux_mol_phase_comp" in var[0].name

    # assert electroneutrality
    m.fs.feed.properties[0].assert_electroneutrality(
        defined_state = True,
        adjust_by_ion = "Cl_-",
        get_property = "flow_mol_phase_comp"
    )
    return m

def fix_init_vars(m):
    # feed state variables
    m.fs.unit.feed_side.properties_in[0].temperature.fix(298.15)
    m.fs.unit.feed_side.properties_in[0].pressure.fix(2e5)
    # pump variables
    m.fs.pump.efficiency_pump[0].fix(0.75)
    m.fs.pump.outlet.pressure[0].fix(2e5)
    # membrane operation
    m.fs.unit.recovery_vol_phase[0,"Liq"].setub(0.95)
    m.fs.unit.spacer_porosity.fix(0.85)
    m.fs.unit.channel_height.fix(5e-4)
    m.fs.unit.velocity[0, 0].fix(0.1)
    m.fs.unit.area.fix(100)
    m.fs.unit.mixed_permeate[0].pressure.fix(101325)
    # variables for calculating mass transfer coefficient with spiral wound correlation
    m.fs.unit.spacer_mixing_efficiency.fix()
    m.fs.unit.spacer_mixing_length.fix()
    # membrane properties
    m.fs.unit.radius_pore.fix(0.5e-9)
    m.fs.unit.membrane_thickness_effective.fix(1.33e-6)
    m.fs.unit.membrane_charge_density.fix(-60)
    m.fs.unit.dielectric_constant_pore.fix(41.3)

def set_NF_feed(
        blk,
        solver,
        flow_mass_h2o,
        conc_mass_phase_comp    # kg/m3
):
    if solver is None:
        solver = get_solver()

    # fix the inlet flow to the block as water flowrate
    mass_flow_in = flow_mass_h2o * pyunits.kg / pyunits.s
    blk.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(flow_mass_h2o)

    # fix the ion cncentrations and unfix ion flows
    for ion, x in conc_mass_phase_comp.items():
        blk.feed.properties[0].conc_mass_phase_comp["Liq", ion].fix(x)
        blk.feed.properties[0].flow_mol_phase_comp["Liq", ion].unfix()
    # solve for the new flow rates
    solver.solve(blk.feed)
    # fix new water concentration
    blk.feed.properties[0].conc_mass_phase_comp["Liq", "H2O"].fix()
    # unfix ion concentrations and fix flows
    for ion, x in conc_mass_phase_comp.items():
        blk.feed.properties[0].conc_mass_phase_comp["Liq", ion].unfix()
        blk.feed.properties[0].flow_mol_phase_comp["Liq", ion].fix()
        #blk.feed.properties[0].flow_mass_phase_comp["Liq", ion].unfix()
    blk.feed.properties[0].conc_mass_phase_comp["Liq", "H2O"].unfix()
    blk.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
    blk.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix()

    # TODO: add and update scaling to a separate function
    # TODO: add electroneutrality here

if __name__ == "__main__":
    main()
