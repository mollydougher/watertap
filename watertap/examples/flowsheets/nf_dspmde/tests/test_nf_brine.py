#################################################################################
# adapted from test_nf.py
#################################################################################

import pytest
from pyomo.environ import value
from watertap.examples.flowsheets.nf_dspmde.nf import main


@pytest.mark.requires_idaes_solver
@pytest.mark.component
def test_main():
    m = main()
    test_dict = {
        "pressure": [m.fs.NF.pump.outlet.pressure[0] * 1e-5, 2.0],
        "area": [m.fs.NF.nfUnit.area, 985.283490],
        "recovery": [
            m.fs.NF.nfUnit.recovery_vol_phase[0.0, "Liq"] * 100,
           9.983259,
        ],
        "ion_ratio": [
            (m.fs.permeate.flow_mol_phase_comp[0,"Liq","Mg_2+"].value/0.024)
            /(m.fs.permeate.flow_mol_phase_comp[0,"Liq","Li_+"].value/0.0069),
            0.493056
        ]
    }
    for model_result, testval in test_dict.values():
        assert pytest.approx(testval, rel=1e-3) == value(model_result)
